# Model Training Application (CLI)

A command-line interface (CLI) application for training machine learning models. This application supports multiple model types including neural networks, linear regression, logistic regression, and random forests.

## Features

- **Multiple Model Types**: Neural Network, Linear Regression, Logistic Regression, Random Forest
- **CSV Dataset Support**: Load and process CSV datasets
- **Progress Tracking**: Real-time training progress with loss and accuracy metrics
- **Model Persistence**: Save and load trained models
- **Cross-platform**: Works on Windows, Linux, and macOS
- **No External Dependencies**: Pure C++ implementation without Qt or other GUI frameworks

## Building the Application

### Prerequisites

- CMake 3.16 or higher
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)

### Windows

```batch
build.bat
```

### Linux/macOS

```bash
chmod +x build.sh
./build.sh
```

## Usage

### Basic Usage

```bash
ModelTrainingApp --dataset <path> [OPTIONS]
```

### Command Line Options

- `--dataset <path>` - Path to the dataset file (CSV format, required)
- `--output <path>` - Output directory for trained model (default: ./models)
- `--model <type>` - Model type: neural, linear, logistic, randomforest (default: neural)
- `--input-size <size>` - Number of input features (default: 8)
- `--hidden-size <size>` - Hidden layer size for neural network (default: 64)
- `--output-size <size>` - Number of output classes (default: 3)
- `--learning-rate <rate>` - Learning rate (default: 0.001)
- `--batch-size <size>` - Batch size (default: 32)
- `--epochs <num>` - Number of training epochs (default: 100)
- `--help` - Show help message

### Examples

Train a neural network with custom parameters:
```bash
ModelTrainingApp --dataset sample_data.csv --output ./models --model neural --epochs 50 --learning-rate 0.01
```

Train a linear regression model:
```bash
ModelTrainingApp --dataset sample_data.csv --model linear --epochs 100
```

Train a random forest:
```bash
ModelTrainingApp --dataset sample_data.csv --model randomforest --epochs 10
```

## Dataset Format

The application expects CSV files with the following format:
- First row: Header (optional)
- Subsequent rows: Feature values followed by label
- Last column: Class label (integer)

Example:
```csv
feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,label
0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0
0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0
0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,1
```

## Model Types

### Neural Network
- Multi-layer perceptron with one hidden layer
- Sigmoid activation function for hidden layer
- Softmax activation for output layer
- Configurable hidden layer size

### Linear Regression
- Standard linear regression model
- Suitable for regression tasks
- Fast training and prediction

### Logistic Regression
- Binary and multi-class classification
- Softmax output for multi-class
- Good baseline for classification tasks

### Random Forest
- Ensemble of decision trees
- Robust to overfitting
- Good for both classification and regression

## Output

The application provides:
- Real-time training progress with progress bar
- Final training metrics (loss, accuracy)
- Training history for the last 10 epochs
- Saved model file in binary format

## Project Structure

```
Model_training/
├── include/           # Header files
│   ├── datasetmanager.h
│   └── modeltrainer.h
├── src/              # Source files
│   ├── main.cpp
│   ├── datasetmanager.cpp
│   └── modeltrainer.cpp
├── sample_data.csv   # Sample dataset
├── CMakeLists.txt    # Build configuration
├── build.bat         # Windows build script
├── build.sh          # Linux/macOS build script
└── README.md         # This file
```

## Technical Details

- **Language**: C++17
- **Build System**: CMake
- **Threading**: Standard C++ threads for training
- **File I/O**: Standard C++ file streams
- **Random Number Generation**: Standard C++ random library
- **Memory Management**: RAII with smart pointers

## Performance

The application is designed for educational and prototyping purposes. For production use with large datasets, consider:
- Using optimized libraries like Eigen or BLAS
- GPU acceleration with CUDA or OpenCL
- More sophisticated optimization algorithms
- Better data preprocessing and augmentation

## License

This project is provided as-is for educational purposes. 