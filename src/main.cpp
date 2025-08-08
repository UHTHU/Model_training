#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <iomanip>
#include <algorithm>
#include <filesystem>

#include "datasetmanager.h"
#include "modeltrainer.h"

void printUsage(const std::string& programName) {
    std::cout << "Usage: " << programName << " [OPTIONS]\n";
    std::cout << "Options:\n";
    std::cout << "  --dataset <path>        Path to the dataset file (CSV format)\n";
    std::cout << "  --output <path>         Output directory for trained model\n";
    std::cout << "  --model <type>          Model type: neural, linear, logistic, randomforest\n";
    std::cout << "  --input-size <size>     Number of input features (default: 8)\n";
    std::cout << "  --hidden-size <size>    Hidden layer size for neural network (default: 64)\n";
    std::cout << "  --output-size <size>    Number of output classes (default: 3)\n";
    std::cout << "  --learning-rate <rate>  Learning rate (default: 0.001)\n";
    std::cout << "  --batch-size <size>     Batch size (default: 32)\n";
    std::cout << "  --epochs <num>          Number of training epochs (default: 100)\n";
    std::cout << "  --help                  Show this help message\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << programName << " --dataset sample_data.csv --output ./models --model neural --epochs 50\n";
}

void printProgress(int epoch, int totalEpochs, double loss, double accuracy) {
    int barWidth = 50;
    float progress = (float)epoch / totalEpochs;
    int pos = barWidth * progress;
    
    std::cout << "\r[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(1) << (progress * 100.0) << "% ";
    std::cout << "Epoch " << epoch << "/" << totalEpochs;
    std::cout << " Loss: " << std::fixed << std::setprecision(4) << loss;
    std::cout << " Acc: " << std::fixed << std::setprecision(4) << accuracy;
    std::cout.flush();
}

void printTrainingSummary(const TrainingMetrics& metrics) {
    std::cout << "\n\n=== Training Summary ===\n";
    std::cout << "Final Loss: " << std::fixed << std::setprecision(4) << metrics.loss << "\n";
    std::cout << "Final Accuracy: " << std::fixed << std::setprecision(4) << metrics.accuracy << "\n";
    std::cout << "Validation Loss: " << std::fixed << std::setprecision(4) << metrics.validationLoss << "\n";
    std::cout << "Validation Accuracy: " << std::fixed << std::setprecision(4) << metrics.validationAccuracy << "\n";
    
    if (!metrics.lossHistory.empty()) {
        std::cout << "\nLoss History (last 10 epochs):\n";
        int start = std::max(0, (int)metrics.lossHistory.size() - 10);
        for (int i = start; i < metrics.lossHistory.size(); ++i) {
            std::cout << "Epoch " << (i + 1) << ": " << std::fixed << std::setprecision(4) 
                      << metrics.lossHistory[i] << "\n";
        }
    }
}

int main(int argc, char* argv[]) {
    std::string datasetPath;
    std::string outputPath = "./models";
    std::string modelType = "neural";
    int inputSize = 8;
    int hiddenSize = 64;
    int outputSize = 3;
    double learningRate = 0.001;
    int batchSize = 32;
    int epochs = 100;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--dataset" && i + 1 < argc) {
            datasetPath = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            modelType = argv[++i];
        } else if (arg == "--input-size" && i + 1 < argc) {
            inputSize = std::stoi(argv[++i]);
        } else if (arg == "--hidden-size" && i + 1 < argc) {
            hiddenSize = std::stoi(argv[++i]);
        } else if (arg == "--output-size" && i + 1 < argc) {
            outputSize = std::stoi(argv[++i]);
        } else if (arg == "--learning-rate" && i + 1 < argc) {
            learningRate = std::stod(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            batchSize = std::stoi(argv[++i]);
        } else if (arg == "--epochs" && i + 1 < argc) {
            epochs = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }
    
    // Validate required parameters
    if (datasetPath.empty()) {
        std::cerr << "Error: Dataset path is required. Use --dataset <path>\n";
        printUsage(argv[0]);
        return 1;
    }
    
    // Check if dataset file exists
    if (!std::filesystem::exists(datasetPath)) {
        std::cerr << "Error: Dataset file not found: " << datasetPath << "\n";
        return 1;
    }
    
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(outputPath);
    
    std::cout << "=== Model Training Application (CLI) ===\n";
    std::cout << "Dataset: " << datasetPath << "\n";
    std::cout << "Output: " << outputPath << "\n";
    std::cout << "Model: " << modelType << "\n";
    std::cout << "Input Size: " << inputSize << "\n";
    std::cout << "Hidden Size: " << hiddenSize << "\n";
    std::cout << "Output Size: " << outputSize << "\n";
    std::cout << "Learning Rate: " << learningRate << "\n";
    std::cout << "Batch Size: " << batchSize << "\n";
    std::cout << "Epochs: " << epochs << "\n\n";
    
    try {
        // Initialize dataset manager
        DatasetManager datasetManager;
        
        // Load dataset
        std::cout << "Loading dataset...\n";
        if (!datasetManager.loadDataset(datasetPath)) {
            std::cerr << "Error: Failed to load dataset\n";
            return 1;
        }
        
        DatasetInfo info = datasetManager.getDatasetInfo();
        std::cout << "Dataset loaded successfully:\n";
        std::cout << "  Total samples: " << info.totalSamples << "\n";
        std::cout << "  Input features: " << info.inputFeatures << "\n";
        std::cout << "  Output classes: " << info.outputClasses << "\n";
        std::cout << "  Format: " << info.format << "\n\n";
        
        // Initialize model trainer
        ModelTrainer trainer;
        trainer.setDatasetPath(datasetPath);
        trainer.setOutputPath(outputPath);
        trainer.setModelType(modelType);
        trainer.setInputSize(inputSize);
        trainer.setHiddenSize(hiddenSize);
        trainer.setOutputSize(outputSize);
        trainer.setLearningRate(learningRate);
        trainer.setBatchSize(batchSize);
        trainer.setEpochs(epochs);
        
        // Set progress callback
        trainer.setProgressCallback([epochs](int epoch, double loss, double accuracy) {
            printProgress(epoch, epochs, loss, accuracy);
        });
        
        // Start training
        std::cout << "Starting training...\n";
        auto startTime = std::chrono::high_resolution_clock::now();
        
        trainer.startTraining();
        
        // Wait for training to complete
        while (trainer.isTraining()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
        
        // Print final results
        printTrainingSummary(trainer.getMetrics());
        
        std::cout << "\nTraining completed in " << duration.count() << " seconds.\n";
        
        // Save the trained model
        std::string modelPath = outputPath + "/trained_model.bin";
        if (trainer.saveModel(modelPath)) {
            std::cout << "Model saved to: " << modelPath << "\n";
        } else {
            std::cerr << "Warning: Failed to save model\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
} 