#include "datasetmanager.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>

DatasetManager::DatasetManager() : datasetLoadedFlag(false) {
    supportedFormats = {"csv", "txt", "bin"};
}

DatasetManager::~DatasetManager() {
    clearDataset();
}

bool DatasetManager::loadDataset(const std::string& path) {
    std::lock_guard<std::mutex> lock(dataMutex);
    
    clearDataset();
    
    datasetInfo.path = path;
    datasetInfo.name = std::filesystem::path(path).filename().string();
    
    // Detect format
    if (!detectDatasetFormat(path)) {
        datasetInfo.errorMessage = "Unsupported file format";
        return false;
    }
    
    // Load based on format
    bool success = false;
    if (datasetInfo.format == "csv") {
        success = loadCSVDataset(path);
    } else if (datasetInfo.format == "txt") {
        success = loadTextDataset(path);
    } else if (datasetInfo.format == "bin") {
        success = loadCustomDataset(path);
    }
    
    if (success) {
        updateDatasetInfo();
        datasetLoadedFlag = true;
    }
    
    return success;
}

bool DatasetManager::validateDataset(const std::string& path) {
    // For now, just check if file exists and is readable
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }
    
    // Basic CSV validation
    std::string line;
    if (std::getline(file, line)) {
        // Count commas to estimate number of features
        int commas = std::count(line.begin(), line.end(), ',');
        if (commas > 0) {
            return true;
        }
    }
    
    return false;
}

DatasetInfo DatasetManager::getDatasetInfo() const {
    std::lock_guard<std::mutex> lock(dataMutex);
    return datasetInfo;
}

std::vector<DataSample> DatasetManager::getTrainingSamples() const {
    std::lock_guard<std::mutex> lock(dataMutex);
    return trainingSamples;
}

std::vector<DataSample> DatasetManager::getValidationSamples() const {
    std::lock_guard<std::mutex> lock(dataMutex);
    return validationSamples;
}

std::vector<DataSample> DatasetManager::getTestSamples() const {
    std::lock_guard<std::mutex> lock(dataMutex);
    return testSamples;
}

std::vector<DataSample> DatasetManager::getSamples(int start, int count) const {
    std::lock_guard<std::mutex> lock(dataMutex);
    
    std::vector<DataSample> result;
    int totalSamples = trainingSamples.size() + validationSamples.size() + testSamples.size();
    
    if (start >= totalSamples || count <= 0) {
        return result;
    }
    
    int end = std::min(start + count, totalSamples);
    
    // Combine all samples
    std::vector<DataSample> allSamples;
    allSamples.insert(allSamples.end(), trainingSamples.begin(), trainingSamples.end());
    allSamples.insert(allSamples.end(), validationSamples.begin(), validationSamples.end());
    allSamples.insert(allSamples.end(), testSamples.begin(), testSamples.end());
    
    result.assign(allSamples.begin() + start, allSamples.begin() + end);
    return result;
}

int DatasetManager::getTotalSamples() const {
    std::lock_guard<std::mutex> lock(dataMutex);
    return trainingSamples.size() + validationSamples.size() + testSamples.size();
}

int DatasetManager::getInputFeatures() const {
    std::lock_guard<std::mutex> lock(dataMutex);
    return datasetInfo.inputFeatures;
}

int DatasetManager::getOutputClasses() const {
    std::lock_guard<std::mutex> lock(dataMutex);
    return datasetInfo.outputClasses;
}

bool DatasetManager::isDatasetLoaded() const {
    std::lock_guard<std::mutex> lock(dataMutex);
    return datasetLoadedFlag;
}

void DatasetManager::clearDataset() {
    trainingSamples.clear();
    validationSamples.clear();
    testSamples.clear();
    datasetLoadedFlag = false;
    datasetInfo = DatasetInfo();
}

bool DatasetManager::loadCSVDataset(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        datasetInfo.errorMessage = "Cannot open file: " + path;
        return false;
    }
    
    std::string line;
    bool firstLine = true;
    int lineCount = 0;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        if (firstLine) {
            // Skip header
            firstLine = false;
            continue;
        }
        
        DataSample sample;
        if (parseCSVLine(line, sample)) {
            // Simple split: 70% training, 15% validation, 15% test
            if (lineCount % 10 < 7) {
                sample.type = "training";
                trainingSamples.push_back(sample);
            } else if (lineCount % 10 < 8) {
                sample.type = "validation";
                validationSamples.push_back(sample);
            } else {
                sample.type = "test";
                testSamples.push_back(sample);
            }
            lineCount++;
        }
    }
    
    if (trainingSamples.empty()) {
        datasetInfo.errorMessage = "No valid data found in CSV file";
        return false;
    }
    
    return true;
}

bool DatasetManager::loadImageDataset(const std::string& path) {
    // Placeholder for image dataset loading
    datasetInfo.errorMessage = "Image dataset loading not implemented";
    return false;
}

bool DatasetManager::loadTextDataset(const std::string& path) {
    // Placeholder for text dataset loading
    datasetInfo.errorMessage = "Text dataset loading not implemented";
    return false;
}

bool DatasetManager::loadCustomDataset(const std::string& path) {
    // Placeholder for custom dataset loading
    datasetInfo.errorMessage = "Custom dataset loading not implemented";
    return false;
}

bool DatasetManager::parseCSVLine(const std::string& line, DataSample& sample) {
    std::istringstream iss(line);
    std::string token;
    std::vector<std::string> tokens;
    
    while (std::getline(iss, token, ',')) {
        tokens.push_back(token);
    }
    
    if (tokens.size() < 2) {
        return false; // Invalid line
    }
    
    // Parse features (all except last token)
    for (size_t i = 0; i < tokens.size() - 1; ++i) {
        try {
            double value = std::stod(tokens[i]);
            sample.features.push_back(value);
        } catch (const std::exception&) {
            // Skip invalid values
            continue;
        }
    }
    
    // Parse label (last token)
    try {
        sample.label = std::stoi(tokens.back());
    } catch (const std::exception&) {
        sample.label = 0; // Default label
    }
    
    return !sample.features.empty(); // Return true if we have at least one feature
}

bool DatasetManager::detectDatasetFormat(const std::string& path) {
    std::string extension = std::filesystem::path(path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    if (extension == ".csv") {
        datasetInfo.format = "csv";
        return true;
    } else if (extension == ".txt") {
        datasetInfo.format = "txt";
        return true;
    } else if (extension == ".bin") {
        datasetInfo.format = "bin";
        return true;
    }
    
    return false;
}

void DatasetManager::updateDatasetInfo() {
    if (!trainingSamples.empty()) {
        datasetInfo.inputFeatures = trainingSamples[0].features.size();
        
        // Find maximum label to determine output classes
        int maxLabel = 0;
        for (const auto& sample : trainingSamples) {
            maxLabel = std::max(maxLabel, sample.label);
        }
        for (const auto& sample : validationSamples) {
            maxLabel = std::max(maxLabel, sample.label);
        }
        for (const auto& sample : testSamples) {
            maxLabel = std::max(maxLabel, sample.label);
        }
        
        datasetInfo.outputClasses = maxLabel + 1; // Labels are 0-based
        datasetInfo.totalSamples = trainingSamples.size() + validationSamples.size() + testSamples.size();
        datasetInfo.isValid = true;
    }
} 