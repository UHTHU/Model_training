#!/bin/bash

echo "Building Model Training Application..."
echo

# Check if CMake is available
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake not found. Please install CMake and add it to your PATH."
    exit 1
fi

# Check if Qt6 is available (try to find qmake)
if ! command -v qmake &> /dev/null; then
    echo "Warning: Qt6 qmake not found in PATH. You may need to set CMAKE_PREFIX_PATH."
    echo
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake ..
if [ $? -ne 0 ]; then
    echo "Error: CMake configuration failed."
    exit 1
fi

# Build the project
echo
echo "Building project..."
make -j$(nproc)
if [ $? -ne 0 ]; then
    echo "Error: Build failed."
    exit 1
fi

echo
echo "Build completed successfully!"
echo
echo "The executable is located at: build/ModelTrainingApp"
echo
echo "To run the application:"
echo "  ./ModelTrainingApp"
echo 