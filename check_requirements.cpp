#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#endif

struct Requirement {
    std::string name;
    std::string description;
    bool (*checker)();
    bool required;
};

// Function declarations
bool checkCppCompiler();
bool checkCMake();
bool checkCpp17Support();
bool checkFilesystemSupport();
bool checkThreadingSupport();
bool checkBuildDirectory();
bool checkSampleData();
std::string getCompilerVersion();
std::string getCMakeVersion();
bool runCommand(const std::string& command, std::string& output);

int main() {
    std::cout << "=== Model Training Application - System Requirements Checker ===\n\n";
    
    std::vector<Requirement> requirements = {
        {"C++ Compiler", "C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)", checkCppCompiler, true},
        {"CMake", "CMake 3.16 or higher", checkCMake, true},
        {"C++17 Support", "C++17 standard library features", checkCpp17Support, true},
        {"Filesystem Support", "std::filesystem support", checkFilesystemSupport, true},
        {"Threading Support", "std::thread and std::mutex support", checkThreadingSupport, true},
        {"Build Directory", "Ability to create build directory", checkBuildDirectory, false},
        {"Sample Data", "Sample dataset file exists", checkSampleData, false}
    };
    
    bool allRequiredPassed = true;
    int passedCount = 0;
    int totalCount = requirements.size();
    
    for (const auto& req : requirements) {
        std::cout << "Checking " << req.name << "... ";
        
        bool result = req.checker();
        
        if (result) {
            std::cout << "✓ PASSED\n";
            passedCount++;
        } else {
            std::cout << "✗ FAILED\n";
            if (req.required) {
                allRequiredPassed = false;
            }
        }
        
        std::cout << "   " << req.description << "\n\n";
    }
    
    // Summary
    std::cout << "=== Summary ===\n";
    std::cout << "Passed: " << passedCount << "/" << totalCount << " checks\n";
    
    if (allRequiredPassed) {
        std::cout << "✓ All required dependencies are satisfied!\n";
        std::cout << "You can now build the application using:\n";
        std::cout << "  Windows: .\\build.bat\n";
        std::cout << "  Linux/Mac: ./build.sh\n";
        std::cout << "\nPress Enter to exit...";
        std::cin.get();
        return 0;
    } else {
        std::cout << "✗ Some required dependencies are missing.\n";
        std::cout << "Please install the missing dependencies and run this checker again.\n";
        std::cout << "\nPress Enter to exit...";
        std::cin.get();
        return 1;
    }
}

bool checkCppCompiler() {
    std::string version;
    
    // Try different compilers
    if (runCommand("g++ --version", version)) {
        std::cout << "   Found GCC: " << version.substr(0, version.find('\n'));
        return true;
    }
    
    if (runCommand("clang++ --version", version)) {
        std::cout << "   Found Clang: " << version.substr(0, version.find('\n'));
        return true;
    }
    
    if (runCommand("cl --version", version)) {
        std::cout << "   Found MSVC: " << version.substr(0, version.find('\n'));
        return true;
    }
    
    // Try MSYS2 specific paths
    if (runCommand("C:\\msys64\\mingw64\\bin\\g++.exe --version", version)) {
        std::cout << "   Found MSYS2 GCC: " << version.substr(0, version.find('\n'));
        return true;
    }
    
    if (runCommand("C:\\msys64\\clang64\\bin\\clang++.exe --version", version)) {
        std::cout << "   Found MSYS2 Clang: " << version.substr(0, version.find('\n'));
        return true;
    }
    
    return false;
}

bool checkCMake() {
    std::string version;
    
    // Try standard CMake
    if (runCommand("cmake --version", version)) {
        // Extract version number
        size_t pos = version.find("cmake version ");
        if (pos != std::string::npos) {
            std::string versionStr = version.substr(pos + 14);
            size_t endPos = versionStr.find('\n');
            if (endPos != std::string::npos) {
                versionStr = versionStr.substr(0, endPos);
            }
            std::cout << "   Found CMake: " << versionStr;
            
            // Parse version
            size_t dot1 = versionStr.find('.');
            size_t dot2 = versionStr.find('.', dot1 + 1);
            if (dot1 != std::string::npos && dot2 != std::string::npos) {
                int major = std::stoi(versionStr.substr(0, dot1));
                int minor = std::stoi(versionStr.substr(dot1 + 1, dot2 - dot1 - 1));
                return major > 3 || (major == 3 && minor >= 16);
            }
        }
    }
    
    // Try MSYS2 CMake
    if (runCommand("C:\\msys64\\mingw64\\bin\\cmake.exe --version", version)) {
        size_t pos = version.find("cmake version ");
        if (pos != std::string::npos) {
            std::string versionStr = version.substr(pos + 14);
            size_t endPos = versionStr.find('\n');
            if (endPos != std::string::npos) {
                versionStr = versionStr.substr(0, endPos);
            }
            std::cout << "   Found MSYS2 CMake: " << versionStr;
            
            // Parse version
            size_t dot1 = versionStr.find('.');
            size_t dot2 = versionStr.find('.', dot1 + 1);
            if (dot1 != std::string::npos && dot2 != std::string::npos) {
                int major = std::stoi(versionStr.substr(0, dot1));
                int minor = std::stoi(versionStr.substr(dot1 + 1, dot2 - dot1 - 1));
                return major > 3 || (major == 3 && minor >= 16);
            }
        }
    }
    
    return false;
}

bool checkCpp17Support() {
    // Create a temporary test file
    std::string testFile = "cpp17_test.cpp";
    std::ofstream file(testFile);
    
    if (!file.is_open()) {
        return false;
    }
    
    // Write C++17 test code
    file << R"(
#include <iostream>
#include <vector>
#include <string>
#include <optional>
#include <variant>
#include <any>
#include <filesystem>

int main() {
    // Test C++17 features
    std::optional<int> opt = 42;
    std::variant<int, std::string> var = "test";
    std::any any_val = 3.14;
    
    // Structured bindings
    std::pair<int, std::string> pair = {1, "hello"};
    auto [first, second] = pair;
    
    // If constexpr
    if constexpr (sizeof(int) == 4) {
        std::cout << "C++17 features work!\n";
    }
    
    return 0;
}
)";
    file.close();
    
    // Try to compile with C++17
    std::string output;
    bool success = false;
    
    if (runCommand("g++ -std=c++17 -o cpp17_test " + testFile, output)) {
        success = true;
    } else if (runCommand("clang++ -std=c++17 -o cpp17_test " + testFile, output)) {
        success = true;
    } else if (runCommand("cl /std:c++17 " + testFile + " /Fe:cpp17_test.exe", output)) {
        success = true;
    } else if (runCommand("C:\\msys64\\mingw64\\bin\\g++.exe -std=c++17 -o cpp17_test " + testFile, output)) {
        success = true;
    } else if (runCommand("C:\\msys64\\clang64\\bin\\clang++.exe -std=c++17 -o cpp17_test " + testFile, output)) {
        success = true;
    }
    
    // Clean up
    std::filesystem::remove(testFile);
    std::filesystem::remove("cpp17_test");
    std::filesystem::remove("cpp17_test.exe");
    
    return success;
}

bool checkFilesystemSupport() {
    // Create a temporary test file
    std::string testFile = "filesystem_test.cpp";
    std::ofstream file(testFile);
    
    if (!file.is_open()) {
        return false;
    }
    
    // Write filesystem test code
    file << R"(
#include <iostream>
#include <filesystem>

int main() {
    std::filesystem::path path("test.txt");
    std::cout << "Filesystem support works!\n";
    return 0;
}
)";
    file.close();
    
    // Try to compile with filesystem support
    std::string output;
    bool success = false;
    
    if (runCommand("g++ -std=c++17 -o filesystem_test " + testFile, output)) {
        success = true;
    } else if (runCommand("clang++ -std=c++17 -o filesystem_test " + testFile, output)) {
        success = true;
    } else if (runCommand("cl /std:c++17 " + testFile + " /Fe:filesystem_test.exe", output)) {
        success = true;
    } else if (runCommand("C:\\msys64\\mingw64\\bin\\g++.exe -std=c++17 -o filesystem_test " + testFile, output)) {
        success = true;
    } else if (runCommand("C:\\msys64\\clang64\\bin\\clang++.exe -std=c++17 -o filesystem_test " + testFile, output)) {
        success = true;
    }
    
    // Clean up
    std::filesystem::remove(testFile);
    std::filesystem::remove("filesystem_test");
    std::filesystem::remove("filesystem_test.exe");
    
    return success;
}

bool checkThreadingSupport() {
    // Create a temporary test file
    std::string testFile = "threading_test.cpp";
    std::ofstream file(testFile);
    
    if (!file.is_open()) {
        return false;
    }
    
    // Write threading test code
    file << R"(
#include <iostream>
#include <thread>
#include <mutex>

int main() {
    std::mutex mtx;
    std::thread t([&mtx]() {
        std::lock_guard<std::mutex> lock(mtx);
        std::cout << "Threading support works!\n";
    });
    t.join();
    return 0;
}
)";
    file.close();
    
    // Try to compile with threading support
    std::string output;
    bool success = false;
    
    if (runCommand("g++ -std=c++17 -pthread -o threading_test " + testFile, output)) {
        success = true;
    } else if (runCommand("clang++ -std=c++17 -pthread -o threading_test " + testFile, output)) {
        success = true;
    } else if (runCommand("cl /std:c++17 " + testFile + " /Fe:threading_test.exe", output)) {
        success = true;
    } else if (runCommand("C:\\msys64\\mingw64\\bin\\g++.exe -std=c++17 -pthread -o threading_test " + testFile, output)) {
        success = true;
    } else if (runCommand("C:\\msys64\\clang64\\bin\\clang++.exe -std=c++17 -pthread -o threading_test " + testFile, output)) {
        success = true;
    }
    
    // Clean up
    std::filesystem::remove(testFile);
    std::filesystem::remove("threading_test");
    std::filesystem::remove("threading_test.exe");
    
    return success;
}

bool checkBuildDirectory() {
    try {
        std::filesystem::create_directories("test_build_dir");
        std::filesystem::remove("test_build_dir");
        return true;
    } catch (...) {
        return false;
    }
}

bool checkSampleData() {
    return std::filesystem::exists("sample_data.csv");
}

std::string getCompilerVersion() {
    std::string output;
    
    if (runCommand("g++ --version", output)) {
        return "GCC";
    } else if (runCommand("clang++ --version", output)) {
        return "Clang";
    } else if (runCommand("cl --version", output)) {
        return "MSVC";
    }
    
    return "Unknown";
}

std::string getCMakeVersion() {
    std::string output;
    
    if (runCommand("cmake --version", output)) {
        size_t pos = output.find("cmake version ");
        if (pos != std::string::npos) {
            return output.substr(pos + 14, output.find('\n', pos) - pos - 14);
        }
    }
    
    return "Unknown";
}

bool runCommand(const std::string& command, std::string& output) {
#ifdef _WIN32
    // Windows implementation
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(sa);
    sa.lpSecurityDescriptor = NULL;
    sa.bInheritHandle = TRUE;
    
    HANDLE hReadPipe, hWritePipe;
    if (!CreatePipe(&hReadPipe, &hWritePipe, &sa, 0)) {
        return false;
    }
    
    STARTUPINFO si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    ZeroMemory(&pi, sizeof(pi));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = hWritePipe;
    si.hStdError = hWritePipe;
    
    std::string cmd = command;
    if (CreateProcess(NULL, &cmd[0], NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
        CloseHandle(hWritePipe);
        
        char buffer[4096];
        DWORD bytesRead;
        output.clear();
        
        while (ReadFile(hReadPipe, buffer, sizeof(buffer) - 1, &bytesRead, NULL) && bytesRead > 0) {
            buffer[bytesRead] = '\0';
            output += buffer;
        }
        
        CloseHandle(hReadPipe);
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
        
        return true;
    }
    
    CloseHandle(hReadPipe);
    CloseHandle(hWritePipe);
    return false;
#else
    // Unix/Linux implementation
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        return false;
    }
    
    char buffer[4096];
    output.clear();
    
    while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
        output += buffer;
    }
    
    int status = pclose(pipe);
    return status == 0;
#endif
}
