@echo off
echo Building Model Training Application (CLI)...

REM Create build directory
if not exist build mkdir build
cd build

REM Configure with CMake
REM Try different generators based on available tools
cmake .. -G "Visual Studio 16 2019" -A x64 2>nul
if %errorlevel% neq 0 (
    echo Visual Studio generator not available, trying MinGW Makefiles...
    cmake .. -G "MinGW Makefiles" 2>nul
    if %errorlevel% neq 0 (
        echo MinGW Makefiles not available, trying Unix Makefiles...
        cmake .. -G "Unix Makefiles" 2>nul
        if %errorlevel% neq 0 (
            echo Error: No suitable CMake generator found.
            echo Please ensure you have Visual Studio, MinGW, or MSYS2 installed.
            pause
            exit /b 1
        )
    )
)

REM Build the project
REM Try different build commands based on generator
cmake --build . --config Release 2>nul
if %errorlevel% neq 0 (
    echo Trying make command for Unix/MinGW generators...
    make 2>nul
    if %errorlevel% neq 0 (
        echo Error: Build failed.
        pause
        exit /b 1
    )
)

echo Build completed!

REM Check for executable in different possible locations
if exist "Release\ModelTrainingApp.exe" (
    echo Executable location: build\Release\ModelTrainingApp.exe
    copy ..\sample_data.csv Release\ 2>nul
) else if exist "ModelTrainingApp.exe" (
    echo Executable location: build\ModelTrainingApp.exe
    copy ..\sample_data.csv . 2>nul
) else (
    echo Warning: Could not find executable in expected location
)



echo.
echo Usage example:
echo ModelTrainingApp.exe --dataset sample_data.csv --output ./models --model neural --epochs 50
echo.
echo For help:
echo ModelTrainingApp.exe --help 