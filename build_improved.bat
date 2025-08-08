@echo off
echo Building Model Training Application (CLI)...
echo.

REM Create build directory
if not exist build mkdir build
cd build

echo Attempting to configure with CMake...
echo.

REM Try different generators in order of preference
REM First try Visual Studio 2022
echo Trying Visual Studio 2022 generator...
cmake .. -G "Visual Studio 17 2022" -A x64
if %errorlevel% equ 0 (
    echo Successfully configured with Visual Studio 2022
    goto :build
)

REM Try Visual Studio 2019
echo Trying Visual Studio 2019 generator...
cmake .. -G "Visual Studio 16 2019" -A x64
if %errorlevel% equ 0 (
    echo Successfully configured with Visual Studio 2019
    goto :build
)

REM Try MinGW Makefiles
echo Trying MinGW Makefiles generator...
cmake .. -G "MinGW Makefiles"
if %errorlevel% equ 0 (
    echo Successfully configured with MinGW Makefiles
    goto :build
)

REM Try Unix Makefiles
echo Trying Unix Makefiles generator...
cmake .. -G "Unix Makefiles"
if %errorlevel% equ 0 (
    echo Successfully configured with Unix Makefiles
    goto :build
)

REM Try NMake Makefiles (default)
echo Trying NMake Makefiles generator...
cmake .. -G "NMake Makefiles"
if %errorlevel% equ 0 (
    echo Successfully configured with NMake Makefiles
    goto :build
)

REM If we get here, no generator worked
echo.
echo Error: No suitable CMake generator found.
echo.
echo Available generators on your system:
cmake --help | findstr "Generators"
echo.
echo Please ensure you have one of the following installed:
echo - Visual Studio 2019 or 2022
echo - MinGW-w64 with g++ compiler
echo - MSYS2 with g++ compiler
echo.
pause
exit /b 1

:build
echo.
echo Building the project...
echo.

REM Try to build using cmake --build first
cmake --build . --config Release
if %errorlevel% equ 0 (
    echo Build completed successfully!
    goto :success
)

REM If cmake --build failed, try make for Unix/MinGW generators
echo Trying make command...
make
if %errorlevel% equ 0 (
    echo Build completed successfully!
    goto :success
)

REM Try mingw32-make for MinGW
echo Trying mingw32-make command...
mingw32-make
if %errorlevel% equ 0 (
    echo Build completed successfully!
    goto :success
)

echo.
echo Error: Build failed with all available methods.
echo Please check the error messages above.
pause
exit /b 1

:success
echo.
echo Checking for executable...

REM Check for executable in different possible locations
if exist "Release\ModelTrainingApp.exe" (
    echo Executable found: build\Release\ModelTrainingApp.exe
    copy ..\sample_data.csv Release\ 2>nul
    echo Sample data copied to build\Release\
) else if exist "ModelTrainingApp.exe" (
    echo Executable found: build\ModelTrainingApp.exe
    copy ..\sample_data.csv . 2>nul
    echo Sample data copied to build\
) else (
    echo Warning: Could not find executable in expected location
    echo Please check the build output for any errors.
)

echo.
echo Build completed successfully!
echo.
echo Usage example:
echo ModelTrainingApp.exe --dataset sample_data.csv --output ./models --model neural --epochs 50
echo.
echo For help:
echo ModelTrainingApp.exe --help
echo.
pause
