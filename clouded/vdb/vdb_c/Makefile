# Makefile for NetCDF to OpenVDB converter
# Compatible with the RAMS build environment

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++14 -O3 -Wall -Wextra

# Library paths - modify these based on your system
NETCDF_DIR = /usr
OPENVDB_DIR = /usr/local
TBB_DIR = /usr
BOOST_DIR = /usr

# Include directories
INCLUDES = -I$(NETCDF_DIR)/include \
           -I$(OPENVDB_DIR)/include \
           -I$(TBB_DIR)/include \
           -I$(BOOST_DIR)/include

# Library directories
LIBDIRS = -L$(NETCDF_DIR)/lib \
          -L$(OPENVDB_DIR)/lib \
          -L$(TBB_DIR)/lib \
          -L$(BOOST_DIR)/lib

# Libraries to link
LIBS = -lnetcdf_c++4 -lnetcdf \
       -lopenvdb \
       -ltbb \
       -lboost_system -lboost_filesystem \
       -lblosc \
       -lHalf \
       -lpthread

# Target executable
TARGET = netcdf_to_vdb
SOURCE = netcdf_to_vdb.cpp

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCE) $(LIBDIRS) $(LIBS)

# Clean target
clean:
	rm -f $(TARGET)

# Install target
install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

# Help target
help:
	@echo "Available targets:"
	@echo "  all      - Build the converter (default)"
	@echo "  clean    - Remove built executable"
	@echo "  install  - Install to /usr/local/bin"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "To customize library paths, edit the variables at the top of this Makefile:"
	@echo "  NETCDF_DIR, OPENVDB_DIR, TBB_DIR, BOOST_DIR"

# Alternative targets for different systems
ubuntu-deps:
	@echo "To install dependencies on Ubuntu/Debian:"
	@echo "sudo apt-get update"
	@echo "sudo apt-get install libnetcdf-dev libnetcdf-c++4-dev"
	@echo "sudo apt-get install libopenvdb-dev libtbb-dev"
	@echo "sudo apt-get install libboost-system-dev libboost-filesystem-dev"
	@echo "sudo apt-get install libblosc-dev libilmbase-dev"

centos-deps:
	@echo "To install dependencies on CentOS/RHEL:"
	@echo "sudo yum install netcdf-devel netcdf-cxx-devel"
	@echo "sudo yum install openvdb-devel tbb-devel"
	@echo "sudo yum install boost-devel"
	@echo "# Note: You may need to build OpenVDB from source on older systems"

.PHONY: all clean install help ubuntu-deps centos-deps