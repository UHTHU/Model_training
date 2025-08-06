# Model Training Application

A comprehensive GUI application for training machine learning models built with Qt6 and C++. This application provides an intuitive interface for loading datasets, configuring model parameters, and monitoring training progress in real-time.

## Features

- **Multiple Model Types**: Support for Neural Networks, Linear Regression, Logistic Regression, and Random Forest
- **Dataset Management**: Load and validate datasets in various formats (CSV, text, images)
- **Real-time Visualization**: Live charts showing training loss and accuracy
- **Model Persistence**: Save and load trained models
- **Progress Monitoring**: Detailed progress tracking with logging
- **Modern UI**: Clean, responsive interface built with Qt6

## Requirements

- **Qt6**: Core, Widgets, and Charts modules
- **CMake**: Version 3.16 or higher
- **C++17**: Compatible compiler (GCC, Clang, MSVC)
- **Windows**: Visual Studio 2019 or later (for Windows builds)

## Building the Application

### Prerequisites

1. **Install Qt6**:
   - Download Qt6 from [qt.io](https://www.qt.io/download)
   - Install Qt6 with the following components:
     - Qt Core
     - Qt Widgets
     - Qt Charts
     - Qt Creator (optional, for development)

2. **Install CMake**:
   - Download from [cmake.org](https://cmake.org/download/)
   - Ensure CMake is in your system PATH

3. **Install a C++ Compiler**:
   - **Windows**: Visual Studio 2019/2022 with C++ workload
   - **Linux**: GCC 7+ or Clang 6+
   - **macOS**: Xcode Command Line Tools

### Build Instructions

1. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd Model_training
   ```

2. **Create build directory**:
   ```bash
   mkdir build
   cd build
   ```

3. **Configure with CMake**:
   ```bash
   # Windows (Visual Studio)
   cmake .. -G "Visual Studio 16 2019" -A x64
   
   # Linux/macOS
   cmake ..
   ```

4. **Build the project**:
   ```bash
   # Windows
   cmake --build . --config Release
   
   # Linux/macOS
   make -j$(nproc)
   ```

5. **Run the application**:
   ```bash
   # Windows
   ./Release/ModelTrainingApp.exe
   
   # Linux/macOS
   ./ModelTrainingApp
   ```

## Usage Guide

### 1. Loading a Dataset

1. Click "Browse" in the Dataset Configuration section
2. Select your dataset file or directory
3. The application will automatically detect the format and display dataset information
4. Review the dataset preview table to verify the data

### 2. Configuring Model Parameters

1. **Model Type**: Choose from Neural Network, Linear Regression, Logistic Regression, or Random Forest
2. **Input Size**: Number of features in your dataset
3. **Hidden Size**: Number of neurons in hidden layer (Neural Network only)
4. **Output Size**: Number of classes or output dimensions
5. **Learning Rate**: Step size for gradient descent (typically 0.001-0.1)
6. **Batch Size**: Number of samples per training batch
7. **Epochs**: Number of complete passes through the dataset

### 3. Starting Training

1. Select an output directory for saving results
2. Click "Start Training"
3. Monitor progress in real-time:
   - Progress bar shows completion percentage
   - Current epoch, loss, and accuracy are displayed
   - Charts update automatically with training curves
   - Log messages provide detailed information

### 4. Saving and Loading Models

- **Save Model**: Click "Save Model" to store the trained model
- **Load Model**: Click "Load Model" to restore a previously saved model

## Supported Dataset Formats

### CSV Files
- Comma-separated values with features and labels
- Last column typically contains the label
- Automatic header detection

### Text Files
- Space or tab-separated values
- Similar format to CSV but with different delimiters

### Image Directories
- Organized by class folders
- Supported formats: JPG, PNG, BMP, TIFF
- Automatic class detection from folder names

## Model Types

### Neural Network
- Multi-layer perceptron with configurable hidden layers
- Sigmoid activation function
- Softmax output for classification
- Gradient descent optimization

### Linear Regression
- Simple linear model for regression tasks
- Mean squared error loss
- Suitable for continuous output prediction

### Logistic Regression
- Binary and multi-class classification
- Cross-entropy loss
- Softmax output for multi-class

### Random Forest
- Ensemble of decision trees
- Robust to overfitting
- Good for both classification and regression

## Project Structure

```
Model_training/
├── CMakeLists.txt          # Main build configuration
├── src/                    # Source files
│   ├── main.cpp           # Application entry point
│   ├── mainwindow.cpp     # Main window implementation
│   ├── datasetmanager.cpp # Dataset handling
│   ├── modeltrainer.cpp   # Model training logic
│   ├── progressdialog.cpp # Progress dialog
│   └── chartwidget.cpp    # Chart visualization
├── include/               # Header files
│   ├── mainwindow.h
│   ├── datasetmanager.h
│   ├── modeltrainer.h
│   ├── progressdialog.h
│   └── chartwidget.h
├── ui/                    # UI files (if using Qt Designer)
├── resources/             # Application resources
│   └── resources.qrc
└── README.md             # This file
```

## Troubleshooting

### Common Build Issues

1. **Qt6 not found**:
   - Ensure Qt6 is properly installed
   - Set `CMAKE_PREFIX_PATH` to Qt6 installation directory
   - Example: `cmake .. -DCMAKE_PREFIX_PATH="C:/Qt/6.5.0/msvc2019_64"`

2. **Compiler not found**:
   - Install appropriate C++ compiler
   - Ensure compiler is in system PATH
   - Use correct CMake generator for your compiler

3. **Missing Qt modules**:
   - Install Qt6 with Charts module
   - Verify all required modules are available

### Runtime Issues

1. **Application crashes on startup**:
   - Check Qt6 DLLs are in PATH (Windows)
   - Verify all dependencies are installed
   - Check console output for error messages

2. **Dataset loading fails**:
   - Verify dataset format is supported
   - Check file permissions
   - Ensure dataset is not corrupted

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Qt6 framework
- Uses Qt Charts for visualization
- Inspired by modern machine learning workflows

## Support

For issues and questions:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed information

---

**Note**: This is a demonstration application. For production use, consider using established machine learning frameworks like TensorFlow, PyTorch, or scikit-learn. 