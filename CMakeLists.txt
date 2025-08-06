cmake_minimum_required(VERSION 3.16)
project(ModelTrainingApp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Qt6 components
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Charts)

# Set up Qt6
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/main.cpp
    src/mainwindow.cpp
    src/datasetmanager.cpp
    src/modeltrainer.cpp
    src/progressdialog.cpp
    src/chartwidget.cpp
)

# Header files
set(HEADERS
    include/mainwindow.h
    include/datasetmanager.h
    include/modeltrainer.h
    include/progressdialog.h
    include/chartwidget.h
)

# UI files
set(UI_FILES
    ui/mainwindow.ui
    ui/progressdialog.ui
)

# Resource files
set(RESOURCES
    resources/resources.qrc
)

# Create executable
add_executable(ModelTrainingApp ${SOURCES} ${HEADERS} ${UI_FILES} ${RESOURCES})

# Link Qt6 libraries
target_link_libraries(ModelTrainingApp 
    Qt6::Core 
    Qt6::Widgets 
    Qt6::Charts
)

# Windows-specific settings
if(WIN32)
    set_target_properties(ModelTrainingApp PROPERTIES
        WIN32_EXECUTABLE TRUE
        OUTPUT_NAME "ModelTrainingApp"
    )
endif()

# Copy Qt DLLs to output directory (for Windows)
if(WIN32)
    add_custom_command(TARGET ModelTrainingApp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:Qt6::Core>
        $<TARGET_FILE:Qt6::Widgets>
        $<TARGET_FILE:Qt6::Charts>
        $<TARGET_FILE_DIR:ModelTrainingApp>
    )
endif() 