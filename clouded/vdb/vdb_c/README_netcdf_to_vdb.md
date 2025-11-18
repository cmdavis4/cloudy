# NetCDF to OpenVDB Converter

A high-performance C++ tool for converting large atmospheric NetCDF files to OpenVDB format, specifically optimized for RAMS (Regional Atmospheric Modeling System) data.

## Features

- **Configurable Variable Selection**: Choose which variables to convert via command line or configuration
- **Large Dataset Support**: Memory-efficient processing for multi-gigabyte atmospheric files
- **Time Series Handling**: Support for multi-timestep data with frame-based output
- **Atmospheric Grid Optimization**: Proper coordinate transforms for meteorological data
- **Data Pruning**: Configurable thresholds to reduce file sizes by removing negligible values
- **RAMS Integration**: Built-in mappings for common RAMS variables and tracers

## Supported Variables

The converter includes built-in mappings for common RAMS atmospheric variables:

### Hydrometeor Variables
- `RCP` → `liquid_water_content`
- `RRP` → `rain_water_content`
- `RIP` → `ice_content`
- `RSP` → `snow_content`
- `RAP` → `aggregates_content`
- `RGP` → `graupel_content`
- `RHP` → `hail_content`

### Meteorological Variables
- `THETA` → `potential_temperature`
- `UP` → `velocity_u`
- `VP` → `velocity_v`
- `WP` → `velocity_w`

### Tracer Variables
- `TRACERP001` through `TRACERP042` → `tracer_1` through `tracer_42`

## Dependencies

### Required Libraries
- **NetCDF C++**: For reading NetCDF files (`libnetcdf-c++4-dev`)
- **OpenVDB**: For writing VDB files (`libopenvdb-dev`)
- **TBB**: Threading Building Blocks (`libtbb-dev`)
- **Boost**: System and filesystem libraries (`libboost-system-dev`, `libboost-filesystem-dev`)
- **Blosc**: Compression library (`libblosc-dev`)
- **Half**: OpenEXR half-precision library (`libilmbase-dev`)

### Ubuntu/Debian Installation
```bash
sudo apt-get update
sudo apt-get install libnetcdf-dev libnetcdf-c++4-dev
sudo apt-get install libopenvdb-dev libtbb-dev
sudo apt-get install libboost-system-dev libboost-filesystem-dev
sudo apt-get install libblosc-dev libilmbase-dev
```

### CentOS/RHEL Installation
```bash
sudo yum install netcdf-devel netcdf-cxx-devel
sudo yum install openvdb-devel tbb-devel boost-devel
# Note: OpenVDB may need to be built from source on older systems
```

## Building

### Option 1: Using CMake (Recommended)
```bash
mkdir build && cd build
cmake ..
make
```

### Option 2: Using Make
```bash
# Edit Makefile to adjust library paths if needed
make
```

### Option 3: Manual Compilation
```bash
g++ -std=c++14 -O3 -Wall -Wextra \
    -I/usr/include -I/usr/local/include \
    -o netcdf_to_vdb netcdf_to_vdb.cpp \
    -L/usr/lib -L/usr/local/lib \
    -lnetcdf_c++4 -lnetcdf -lopenvdb -ltbb \
    -lboost_system -lboost_filesystem -lblosc -lHalf -lpthread
```

## Usage

### Basic Usage
```bash
./netcdf_to_vdb input.nc output.vdb
```

### Variable Selection
```bash
# Convert specific variables
./netcdf_to_vdb storm_data.nc cloud_volume.vdb -v RCP,TRACERP001,THETA

# Convert all tracer variables
./netcdf_to_vdb data.nc tracers.vdb -v TRACERP001,TRACERP002,TRACERP003
```

### Time Series Processing
```bash
# Convert all timesteps (creates multiple output files)
./netcdf_to_vdb timeseries.nc animation.vdb --verbose

# Convert specific timestep
./netcdf_to_vdb timeseries.nc frame_010.vdb -T 10
```

### Advanced Options
```bash
# High-resolution data with custom threshold
./netcdf_to_vdb highres.nc output.vdb -s 250.0 -t 1e-8 --verbose

# Batch processing with shell script
for file in *.nc; do
    ./netcdf_to_vdb "$file" "${file%.nc}.vdb" -v RCP,RRP,THETA
done
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-v, --variables` | Comma-separated list of variables | All available |
| `-t, --threshold` | Pruning threshold for small values | 1e-6 |
| `-s, --spacing` | Grid spacing in meters | 500.0 |
| `-T, --timestep` | Specific timestep (-1 for all) | -1 |
| `--verbose` | Enable detailed output | false |
| `-h, --help` | Show help message | - |

## File Naming Conventions

### Single Timestep
```
input.nc → output.vdb
```

### Multiple Timesteps
```
input.nc → output_t0.vdb, output_t1.vdb, output_t2.vdb, ...
```

## Performance Optimization

### Memory Usage
- The converter processes data in chunks to handle large files
- Use appropriate thresholds to reduce output file sizes
- Consider converting subsets of variables for very large datasets

### Typical Processing Times
- **Small files** (< 1 GB): Seconds to minutes
- **Large files** (1-10 GB): Minutes to hours
- **Very large files** (> 10 GB): Hours, consider parallel processing

### Recommended Workflow
1. **Test with small dataset**: Verify variable selection and output quality
2. **Optimize threshold**: Balance file size vs. data preservation
3. **Batch process**: Use shell scripts for multiple files
4. **Monitor resources**: Watch memory and disk usage for large conversions

## Integration with RAMS Workflow

### Typical RAMS to Blender Pipeline
```bash
# 1. Run RAMS simulation (generates NetCDF output)
cd RAMS_base/bin.rams && ./rams

# 2. Convert to VDB for visualization
cd /path/to/bl_transport
./netcdf_to_vdb /moonbow/cmdavis4/projects/bl_transport/rams_io/storm_ic.nc storm_ic.vdb \
    -v RCP,TRACERP001,TRACERP002 --verbose

# 3. Import VDB files into Blender for rendering
```

### Configuration for Different Storm Types

#### Isolated Convection
```bash
./netcdf_to_vdb storm_ic.nc ic_clouds.vdb -v RCP,RRP,RIP -t 1e-5
```

#### Squall Line
```bash
./netcdf_to_vdb storm_sl.nc sl_structure.vdb -v RCP,THETA,UP,VP,WP -s 500
```

#### Supercell
```bash
./netcdf_to_vdb storm_sc.nc sc_tracers.vdb -v TRACERP001,TRACERP007,TRACERP015 --verbose
```

## Troubleshooting

### Common Issues

1. **"Error: Could not determine spatial dimensions"**
   - Check that NetCDF file has recognizable dimension names (x/xi, y/yi, z/zi)
   - Use `ncdump -h file.nc` to inspect dimensions

2. **"Error: No convertible variables found"**
   - List available variables with `--verbose` flag
   - Specify variables manually with `-v` option

3. **Compilation errors about missing libraries**
   - Verify all dependencies are installed
   - Check library paths in Makefile
   - Use `pkg-config --libs netcdf` to find correct linking flags

4. **Very large output files**
   - Increase pruning threshold with `-t` option
   - Convert fewer variables at once
   - Check for unrealistic data values in source

5. **Memory issues with large files**
   - Process timesteps individually with `-T` option
   - Reduce grid resolution if possible
   - Monitor system memory usage

### Debug Mode
Use `--verbose` flag to see detailed processing information:
```bash
./netcdf_to_vdb input.nc output.vdb --verbose
```

## Contributing

When modifying the converter:

1. **Test with sample data**: Use small RAMS output files first
2. **Verify output**: Load VDB files in Blender to check correctness
3. **Performance testing**: Monitor memory usage with large files
4. **Documentation**: Update this README for new features

## Examples

See the `examples/` directory for sample configuration files and conversion scripts for different atmospheric scenarios.

## License

This tool is designed for use with the RAMS atmospheric modeling system and follows the same licensing terms as the broader bl_transport project.