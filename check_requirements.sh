#!/bin/bash

echo "Building System Requirements Checker..."

# Try to compile the checker
echo "Compiling requirements checker..."

# Try different compilers
if g++ -std=c++17 -o check_requirements check_requirements.cpp 2>/dev/null; then
    echo "Compilation successful with GCC"
elif clang++ -std=c++17 -o check_requirements check_requirements.cpp 2>/dev/null; then
    echo "Compilation successful with Clang"
else
    echo "Error: Could not compile the requirements checker."
    echo "Please ensure you have a C++17 compatible compiler installed."
    exit 1
fi

echo ""
echo "Running system requirements check..."
echo ""

# Run the checker
./check_requirements

# Clean up
rm -f check_requirements
