cmake_minimum_required(VERSION 3.16)
project(ModelTrainingApp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/main.cpp
    src/datasetmanager.cpp
    src/modeltrainer.cpp
)

# Header files
set(HEADERS
    include/datasetmanager.h
    include/modeltrainer.h
)

# Create executable
add_executable(ModelTrainingApp ${SOURCES} ${HEADERS})

# Windows-specific settings
if(WIN32)
    set_target_properties(ModelTrainingApp PROPERTIES
        OUTPUT_NAME "ModelTrainingApp"
    )
endif() 