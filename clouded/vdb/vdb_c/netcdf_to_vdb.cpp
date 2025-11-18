/**
 * NetCDF to OpenVDB Converter
 * 
 * Converts large atmospheric NetCDF files to OpenVDB format with configurable
 * variable selection, optimized for RAMS atmospheric modeling data.
 * 
 * Usage: ./netcdf_to_vdb input.nc output.vdb [options]
 */

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <sstream>
#include <memory>
#include <cmath>

// NetCDF includes
#include <netcdf>

// OpenVDB includes
#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>
#include <openvdb/tools/GridTransformer.h>

// Command line argument parsing
#include <getopt.h>

// JSON configuration support
#include <fstream>

class NetCDFToVDBConverter {
private:
    std::string input_file;
    std::string output_file;
    std::vector<std::string> variables;
    std::map<std::string, std::string> variable_mapping;
    double threshold = 1e-6;
    double grid_spacing = 500.0; // Default 500m for RAMS
    bool verbose = false;
    int time_step = -1; // -1 means all timesteps

public:
    NetCDFToVDBConverter(const std::string& input, const std::string& output) 
        : input_file(input), output_file(output) {
        
        // Default variable mappings for common RAMS variables
        variable_mapping["RCP"] = "liquid_water_content";
        variable_mapping["RRP"] = "rain_water_content";
        variable_mapping["RIP"] = "ice_content";
        variable_mapping["RSP"] = "snow_content";
        variable_mapping["RAP"] = "aggregates_content";
        variable_mapping["RGP"] = "graupel_content";
        variable_mapping["RHP"] = "hail_content";
        variable_mapping["THETA"] = "potential_temperature";
        variable_mapping["UP"] = "velocity_u";
        variable_mapping["VP"] = "velocity_v";
        variable_mapping["WP"] = "velocity_w";
        
        // Add tracer mappings
        for (int i = 1; i <= 42; ++i) {
            char var_name[20];
            sprintf(var_name, "TRACERP%03d", i);
            variable_mapping[var_name] = std::string("tracer_") + std::to_string(i);
        }
    }

    void setVariables(const std::vector<std::string>& vars) {
        variables = vars;
    }

    void setThreshold(double thresh) {
        threshold = thresh;
    }

    void setGridSpacing(double spacing) {
        grid_spacing = spacing;
    }

    void setVerbose(bool v) {
        verbose = v;
    }

    void setTimeStep(int step) {
        time_step = step;
    }

    bool convert() {
        try {
            // Initialize OpenVDB
            openvdb::initialize();

            // Open NetCDF file
            netCDF::NcFile ncFile(input_file, netCDF::NcFile::read);
            
            if (verbose) {
                std::cout << "Reading NetCDF file: " << input_file << std::endl;
                std::cout << "Variables in file:" << std::endl;
                auto vars = ncFile.getVars();
                for (const auto& var : vars) {
                    std::cout << "  " << var.first << std::endl;
                }
            }

            // Get dimensions
            auto dims = ncFile.getDims();
            int nx = 0, ny = 0, nz = 0, nt = 0;
            
            // Try to find spatial dimensions (common RAMS naming)
            if (dims.find("x") != dims.end()) nx = dims["x"].getSize();
            else if (dims.find("xi") != dims.end()) nx = dims["xi"].getSize();
            else if (dims.find("lon") != dims.end()) nx = dims["lon"].getSize();
            
            if (dims.find("y") != dims.end()) ny = dims["y"].getSize();
            else if (dims.find("yi") != dims.end()) ny = dims["yi"].getSize();
            else if (dims.find("lat") != dims.end()) ny = dims["lat"].getSize();
            
            if (dims.find("z") != dims.end()) nz = dims["z"].getSize();
            else if (dims.find("zi") != dims.end()) nz = dims["zi"].getSize();
            else if (dims.find("level") != dims.end()) nz = dims["level"].getSize();
            
            if (dims.find("time") != dims.end()) nt = dims["time"].getSize();
            else if (dims.find("t") != dims.end()) nt = dims["t"].getSize();
            
            if (verbose) {
                std::cout << "Grid dimensions: " << nx << " x " << ny << " x " << nz;
                if (nt > 0) std::cout << " (time steps: " << nt << ")";
                std::cout << std::endl;
            }

            if (nx == 0 || ny == 0 || nz == 0) {
                std::cerr << "Error: Could not determine spatial dimensions" << std::endl;
                return false;
            }

            // If no variables specified, use all available that we can map
            if (variables.empty()) {
                auto vars = ncFile.getVars();
                for (const auto& var : vars) {
                    if (variable_mapping.find(var.first) != variable_mapping.end()) {
                        variables.push_back(var.first);
                    }
                }
            }

            if (variables.empty()) {
                std::cerr << "Error: No convertible variables found" << std::endl;
                return false;
            }

            // Determine time steps to process
            std::vector<int> time_steps;
            if (time_step >= 0 && time_step < nt) {
                time_steps.push_back(time_step);
            } else if (nt > 0) {
                for (int t = 0; t < nt; ++t) {
                    time_steps.push_back(t);
                }
            } else {
                time_steps.push_back(0); // Single timestep
            }

            // Process each time step
            for (int t : time_steps) {
                std::string output_filename = output_file;
                if (nt > 1) {
                    // Add timestep to filename
                    size_t pos = output_filename.find_last_of('.');
                    if (pos != std::string::npos) {
                        output_filename = output_filename.substr(0, pos) + 
                                        "_t" + std::to_string(t) + 
                                        output_filename.substr(pos);
                    } else {
                        output_filename += "_t" + std::to_string(t);
                    }
                }

                if (verbose) {
                    std::cout << "Processing time step " << t << " -> " << output_filename << std::endl;
                }

                convertTimeStep(ncFile, t, nx, ny, nz, output_filename);
            }

            ncFile.close();
            return true;

        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return false;
        }
    }

private:
    void convertTimeStep(netCDF::NcFile& ncFile, int timestep, 
                        int nx, int ny, int nz, const std::string& output_filename) {
        
        // Create transform for the grid
        openvdb::math::Transform::Ptr transform = 
            openvdb::math::Transform::createLinearTransform(grid_spacing);

        // Create grids collection
        openvdb::GridPtrVec grids;

        // Process each variable
        for (const std::string& var_name : variables) {
            auto ncVar = ncFile.getVar(var_name);
            if (ncVar.isNull()) {
                if (verbose) {
                    std::cout << "Warning: Variable " << var_name << " not found, skipping" << std::endl;
                }
                continue;
            }

            if (verbose) {
                std::cout << "  Processing variable: " << var_name << std::endl;
            }

            // Read the data
            std::vector<float> data(nx * ny * nz);
            
            // Handle different dimensionalities
            auto dims = ncVar.getDims();
            if (dims.size() == 4) {
                // 4D: (time, z, y, x)
                std::vector<size_t> start = {static_cast<size_t>(timestep), 0, 0, 0};
                std::vector<size_t> count = {1, static_cast<size_t>(nz), 
                                           static_cast<size_t>(ny), static_cast<size_t>(nx)};
                ncVar.getVar(start, count, data.data());
            } else if (dims.size() == 3) {
                // 3D: (z, y, x)
                std::vector<size_t> start = {0, 0, 0};
                std::vector<size_t> count = {static_cast<size_t>(nz), 
                                           static_cast<size_t>(ny), static_cast<size_t>(nx)};
                ncVar.getVar(start, count, data.data());
            } else {
                if (verbose) {
                    std::cout << "Warning: Unsupported dimensionality for " << var_name << std::endl;
                }
                continue;
            }

            // Create OpenVDB grid
            auto grid = createVDBGrid(data, nx, ny, nz, transform, var_name);
            if (grid) {
                grids.push_back(grid);
            }
        }

        // Write the grids to file
        if (!grids.empty()) {
            openvdb::io::File file(output_filename);
            file.write(grids);
            file.close();
            
            if (verbose) {
                std::cout << "  Wrote " << grids.size() << " grids to " << output_filename << std::endl;
            }
        }
    }

    openvdb::FloatGrid::Ptr createVDBGrid(const std::vector<float>& data, 
                                         int nx, int ny, int nz,
                                         openvdb::math::Transform::Ptr transform,
                                         const std::string& var_name) {
        
        // Create the grid
        auto grid = openvdb::FloatGrid::create();
        
        // Set the name
        std::string grid_name = var_name;
        if (variable_mapping.find(var_name) != variable_mapping.end()) {
            grid_name = variable_mapping[var_name];
        }
        grid->setName(grid_name);
        
        // Set transform
        grid->setTransform(transform);

        // Use Dense to populate the grid efficiently
        openvdb::tools::Dense<float> dense(openvdb::Coord(nx, ny, nz));
        
        // Copy data with proper ordering: NetCDF (Z,Y,X) -> OpenVDB (X,Y,Z)
        // This is equivalent to the Python transpose (2,1,0) operation
        float* dense_data = dense.data();
        for (int k = 0; k < nz; ++k) {         // Z (vertical levels)
            for (int j = 0; j < ny; ++j) {     // Y (north-south)
                for (int i = 0; i < nx; ++i) { // X (east-west)
                    // NetCDF index: (z,y,x) order
                    int nc_idx = k * nx * ny + j * nx + i;
                    // OpenVDB index: (x,y,z) order - transpose to get correct orientation
                    int vdb_idx = i * ny * nz + j * nz + k;
                    dense_data[vdb_idx] = data[nc_idx];
                }
            }
        }

        // Copy from dense to sparse grid
        openvdb::tools::copyFromDense(dense, *grid, static_cast<float>(threshold));

        // Additional pruning
        grid->pruneGrid(static_cast<float>(threshold));

        if (verbose) {
            std::cout << "    " << grid_name << ": " << grid->activeVoxelCount() 
                      << " active voxels" << std::endl;
        }

        return grid;
    }
};

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " input.nc output.vdb [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -v, --variables VAR1,VAR2,...  Comma-separated list of variables to convert\n";
    std::cout << "  -t, --threshold THRESHOLD      Threshold for pruning small values (default: 1e-6)\n";
    std::cout << "  -s, --spacing SPACING          Grid spacing in meters (default: 500.0)\n";
    std::cout << "  -T, --timestep STEP            Convert only specific timestep (-1 for all, default: -1)\n";
    std::cout << "  --verbose                      Enable verbose output\n";
    std::cout << "  -h, --help                     Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " storm_ic.nc storm_ic.vdb\n";
    std::cout << "  " << program_name << " data.nc output.vdb -v RCP,TRACERP001,THETA --verbose\n";
    std::cout << "  " << program_name << " data.nc frame.vdb -T 10 -t 1e-5\n";
}

std::vector<std::string> splitString(const std::string& str, char delimiter) {
    std::vector<std::string> result;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, delimiter)) {
        if (!item.empty()) {
            result.push_back(item);
        }
    }
    return result;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];
    
    NetCDFToVDBConverter converter(input_file, output_file);

    // Parse command line options
    static struct option long_options[] = {
        {"variables", required_argument, 0, 'v'},
        {"threshold", required_argument, 0, 't'},
        {"spacing", required_argument, 0, 's'},
        {"timestep", required_argument, 0, 'T'},
        {"verbose", no_argument, 0, 1},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int option_index = 0;
    int c;
    
    while ((c = getopt_long(argc, argv, "v:t:s:T:h", long_options, &option_index)) != -1) {
        switch (c) {
            case 'v': {
                auto vars = splitString(optarg, ',');
                converter.setVariables(vars);
                break;
            }
            case 't':
                converter.setThreshold(std::atof(optarg));
                break;
            case 's':
                converter.setGridSpacing(std::atof(optarg));
                break;
            case 'T':
                converter.setTimeStep(std::atoi(optarg));
                break;
            case 1: // --verbose
                converter.setVerbose(true);
                break;
            case 'h':
                printUsage(argv[0]);
                return 0;
            case '?':
                // getopt_long already printed an error message
                return 1;
            default:
                abort();
        }
    }

    // Perform the conversion
    if (converter.convert()) {
        std::cout << "Conversion completed successfully!" << std::endl;
        return 0;
    } else {
        std::cerr << "Conversion failed!" << std::endl;
        return 1;
    }
}