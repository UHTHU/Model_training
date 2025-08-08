@echo off
echo Building Model Training Application (CLI)...

REM Create build directory
if not exist build mkdir build
cd build

REM Configure with CMake
cmake .. -G "Visual Studio 16 2019" -A x64

REM Build the project
cmake --build . --config Release

echo Build completed!
echo Executable location: build\Release\ModelTrainingApp.exe

REM Copy sample data to build directory
copy ..\sample_data.csv Release\

echo.
echo Usage example:
echo ModelTrainingApp.exe --dataset sample_data.csv --output ./models --model neural --epochs 50
echo.
echo For help:
echo ModelTrainingApp.exe --help 