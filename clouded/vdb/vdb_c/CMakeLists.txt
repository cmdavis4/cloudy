cmake_minimum_required(VERSION 3.12)
project(NetCDFToVDB)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# Find NetCDF
find_package(netCDF REQUIRED)
if(NOT netCDF_FOUND)
    # Try with pkg-config as fallback
    pkg_check_modules(NetCDF REQUIRED netcdf)
endif()

# Find OpenVDB
find_package(OpenVDB REQUIRED)
if(NOT OpenVDB_FOUND)
    # Manual search for OpenVDB
    find_path(OpenVDB_INCLUDE_DIR 
        NAMES openvdb/openvdb.h
        PATHS /usr/include /usr/local/include /opt/local/include
    )
    find_library(OpenVDB_LIBRARY
        NAMES openvdb
        PATHS /usr/lib /usr/local/lib /opt/local/lib
    )
    if(OpenVDB_INCLUDE_DIR AND OpenVDB_LIBRARY)
        set(OpenVDB_FOUND TRUE)
        set(OpenVDB_INCLUDE_DIRS ${OpenVDB_INCLUDE_DIR})
        set(OpenVDB_LIBRARIES ${OpenVDB_LIBRARY})
    endif()
endif()

# Find TBB (required by OpenVDB)
find_package(TBB REQUIRED)
if(NOT TBB_FOUND)
    pkg_check_modules(TBB REQUIRED tbb)
endif()

# Find Blosc (compression library used by OpenVDB)
find_library(Blosc_LIBRARY
    NAMES blosc
    PATHS /usr/lib /usr/local/lib /opt/local/lib
)

# Find Half (OpenEXR)
find_library(Half_LIBRARY
    NAMES Half
    PATHS /usr/lib /usr/local/lib /opt/local/lib
)

# Find Boost (required by OpenVDB)
find_package(Boost REQUIRED COMPONENTS system filesystem)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

# Include directories
include_directories(${netCDF_INCLUDE_DIRS})
include_directories(${OpenVDB_INCLUDE_DIRS})
include_directories(${TBB_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

# Create the executable
add_executable(netcdf_to_vdb netcdf_to_vdb.cpp)

# Link libraries
target_link_libraries(netcdf_to_vdb 
    ${netCDF_LIBRARIES}
    ${OpenVDB_LIBRARIES}
    ${TBB_LIBRARIES}
    ${Boost_LIBRARIES}
    ${Blosc_LIBRARY}
    ${Half_LIBRARY}
    -lpthread
)

# Additional linker flags for NetCDF
if(NetCDF_LIBRARY_DIRS)
    target_link_directories(netcdf_to_vdb PRIVATE ${NetCDF_LIBRARY_DIRS})
endif()

# Set RPATH for shared libraries
set_target_properties(netcdf_to_vdb PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Install target
install(TARGETS netcdf_to_vdb DESTINATION bin)

# Print configuration summary
message(STATUS "NetCDF include dirs: ${netCDF_INCLUDE_DIRS}")
message(STATUS "NetCDF libraries: ${netCDF_LIBRARIES}")
message(STATUS "OpenVDB include dirs: ${OpenVDB_INCLUDE_DIRS}")
message(STATUS "OpenVDB libraries: ${OpenVDB_LIBRARIES}")
message(STATUS "TBB libraries: ${TBB_LIBRARIES}")
message(STATUS "Boost libraries: ${Boost_LIBRARIES}")