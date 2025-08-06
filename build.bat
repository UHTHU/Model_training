@echo off
echo Building Model Training Application...
echo.

REM Check if CMake is available
cmake --version >nul 2>&1
if errorlevel 1 (
    echo Error: CMake not found. Please install CMake and add it to your PATH.
    pause
    exit /b 1
)

REM Check if Qt6 is available (try to find qmake)
where qmake >nul 2>&1
if errorlevel 1 (
    echo Warning: Qt6 qmake not found in PATH. You may need to set CMAKE_PREFIX_PATH.
    echo.
)

REM Create build directory
if not exist build mkdir build
cd build

REM Configure with CMake
echo Configuring with CMake...
cmake .. -G "Visual Studio 16 2019" -A x64
if errorlevel 1 (
    echo Error: CMake configuration failed.
    pause
    exit /b 1
)

REM Build the project
echo.
echo Building project...
cmake --build . --config Release
if errorlevel 1 (
    echo Error: Build failed.
    pause
    exit /b 1
)

echo.
echo Build completed successfully!
echo.
echo The executable is located at: build\Release\ModelTrainingApp.exe
echo.
echo To run the application:
echo   cd build\Release
echo   ModelTrainingApp.exe
echo.

pause 