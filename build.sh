#!/bin/bash

echo "Building Model Training Application (CLI)..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build the project
make -j$(nproc)

echo "Build completed!"
echo "Executable location: build/ModelTrainingApp"

# Copy sample data to build directory
cp ../sample_data.csv .

echo ""
echo "Usage example:"
echo "./ModelTrainingApp --dataset sample_data.csv --output ./models --model neural --epochs 50"
echo ""
echo "For help:"
echo "./ModelTrainingApp --help" 