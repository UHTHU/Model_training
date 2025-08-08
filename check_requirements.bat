@echo off
echo Building System Requirements Checker...

REM Try to compile the checker
echo Compiling requirements checker...

REM Try different compilers
g++ -std=c++17 -o check_requirements.exe check_requirements.cpp 2>nul
if %errorlevel% equ 0 (
    echo Compilation successful with GCC
    goto run_checker
)

clang++ -std=c++17 -o check_requirements.exe check_requirements.cpp 2>nul
if %errorlevel% equ 0 (
    echo Compilation successful with Clang
    goto run_checker
)

cl /std:c++17 check_requirements.cpp /Fe:check_requirements.exe 2>nul
if %errorlevel% equ 0 (
    echo Compilation successful with MSVC
    goto run_checker
)

REM Try MSYS2 compilers
C:\msys64\mingw64\bin\g++.exe -std=c++17 -o check_requirements.exe check_requirements.cpp 2>nul
if %errorlevel% equ 0 (
    echo Compilation successful with MSYS2 GCC
    goto run_checker
)

C:\msys64\clang64\bin\clang++.exe -std=c++17 -o check_requirements.exe check_requirements.cpp 2>nul
if %errorlevel% equ 0 (
    echo Compilation successful with MSYS2 Clang
    goto run_checker
)

echo Error: Could not compile the requirements checker.
echo Please ensure you have a C++17 compatible compiler installed.
pause
exit /b 1

:run_checker
echo.
echo Running system requirements check...
echo.
check_requirements.exe

REM Clean up
del check_requirements.exe 2>nul
