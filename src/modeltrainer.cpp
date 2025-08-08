#include "modeltrainer.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>

// Neural Network Implementation
NeuralNetwork::NeuralNetwork() : inputSize(0), hiddenSize(64), outputSize(0) {}

NeuralNetwork::~NeuralNetwork() {}

void NeuralNetwork::initialize(int inputSize, int outputSize) {
    this->inputSize = inputSize;
    this->outputSize = outputSize;
    
    // Initialize weights with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.1);
    
    // Initialize weights1 (input to hidden)
    weights1.resize(inputSize);
    for (auto& row : weights1) {
        row.resize(hiddenSize);
        for (auto& weight : row) {
            weight = dist(gen);
        }
    }
    
    // Initialize weights2 (hidden to output)
    weights2.resize(hiddenSize);
    for (auto& row : weights2) {
        row.resize(outputSize);
        for (auto& weight : row) {
            weight = dist(gen);
        }
    }
    
    // Initialize biases
    bias1.resize(hiddenSize);
    bias2.resize(outputSize);
    for (auto& b : bias1) b = dist(gen);
    for (auto& b : bias2) b = dist(gen);
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
    if (input.size() != inputSize) {
        return std::vector<double>();
    }
    
    // Forward pass through hidden layer
    std::vector<double> hidden(hiddenSize);
    for (int j = 0; j < hiddenSize; ++j) {
        hidden[j] = bias1[j];
        for (int i = 0; i < inputSize; ++i) {
            hidden[j] += input[i] * weights1[i][j];
        }
        hidden[j] = sigmoid(hidden[j]);
    }
    
    // Forward pass through output layer
    std::vector<double> output(outputSize);
    for (int k = 0; k < outputSize; ++k) {
        output[k] = bias2[k];
        for (int j = 0; j < hiddenSize; ++j) {
            output[k] += hidden[j] * weights2[j][k];
        }
    }
    
    return softmax(output);
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& inputs, 
                         const std::vector<std::vector<double>>& targets, 
                         double learningRate) {
    if (inputs.empty() || inputs.size() != targets.size()) {
        return;
    }
    
    int batchSize = inputs.size();
    
    // Forward pass
    std::vector<std::vector<double>> hiddenOutputs(batchSize, std::vector<double>(hiddenSize));
    std::vector<std::vector<double>> finalOutputs(batchSize, std::vector<double>(outputSize));
    
    for (int b = 0; b < batchSize; ++b) {
        // Hidden layer
        for (int j = 0; j < hiddenSize; ++j) {
            hiddenOutputs[b][j] = bias1[j];
            for (int i = 0; i < inputSize; ++i) {
                hiddenOutputs[b][j] += inputs[b][i] * weights1[i][j];
            }
            hiddenOutputs[b][j] = sigmoid(hiddenOutputs[b][j]);
        }
        
        // Output layer
        for (int k = 0; k < outputSize; ++k) {
            finalOutputs[b][k] = bias2[k];
            for (int j = 0; j < hiddenSize; ++j) {
                finalOutputs[b][k] += hiddenOutputs[b][j] * weights2[j][k];
            }
        }
        finalOutputs[b] = softmax(finalOutputs[b]);
    }
    
    // Backward pass (simplified)
    // This is a simplified backpropagation - in a real implementation,
    // you would compute gradients properly
    for (int b = 0; b < batchSize; ++b) {
        for (int k = 0; k < outputSize; ++k) {
            double error = targets[b][k] - finalOutputs[b][k];
            bias2[k] += learningRate * error;
            
            for (int j = 0; j < hiddenSize; ++j) {
                weights2[j][k] += learningRate * error * hiddenOutputs[b][j];
            }
        }
        
        for (int j = 0; j < hiddenSize; ++j) {
            double error = 0;
            for (int k = 0; k < outputSize; ++k) {
                error += (targets[b][k] - finalOutputs[b][k]) * weights2[j][k];
            }
            error *= sigmoidDerivative(hiddenOutputs[b][j]);
            
            bias1[j] += learningRate * error;
            for (int i = 0; i < inputSize; ++i) {
                weights1[i][j] += learningRate * error * inputs[b][i];
            }
        }
    }
}

bool NeuralNetwork::save(const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    // Save architecture
    file.write(reinterpret_cast<const char*>(&inputSize), sizeof(inputSize));
    file.write(reinterpret_cast<const char*>(&hiddenSize), sizeof(hiddenSize));
    file.write(reinterpret_cast<const char*>(&outputSize), sizeof(outputSize));
    
    // Save weights and biases
    for (const auto& row : weights1) {
        file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
    }
    for (const auto& row : weights2) {
        file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
    }
    file.write(reinterpret_cast<const char*>(bias1.data()), bias1.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(bias2.data()), bias2.size() * sizeof(double));
    
    return true;
}

bool NeuralNetwork::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    // Load architecture
    file.read(reinterpret_cast<char*>(&inputSize), sizeof(inputSize));
    file.read(reinterpret_cast<char*>(&hiddenSize), sizeof(hiddenSize));
    file.read(reinterpret_cast<char*>(&outputSize), sizeof(outputSize));
    
    // Load weights and biases
    weights1.resize(inputSize);
    for (auto& row : weights1) {
        row.resize(hiddenSize);
        file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
    }
    
    weights2.resize(hiddenSize);
    for (auto& row : weights2) {
        row.resize(outputSize);
        file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
    }
    
    bias1.resize(hiddenSize);
    bias2.resize(outputSize);
    file.read(reinterpret_cast<char*>(bias1.data()), bias1.size() * sizeof(double));
    file.read(reinterpret_cast<char*>(bias2.data()), bias2.size() * sizeof(double));
    
    return true;
}

void NeuralNetwork::setHiddenSize(int size) {
    hiddenSize = size;
}

double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double NeuralNetwork::sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

std::vector<double> NeuralNetwork::softmax(const std::vector<double>& input) {
    std::vector<double> output = input;
    double maxVal = *std::max_element(output.begin(), output.end());
    
    double sum = 0.0;
    for (auto& val : output) {
        val = std::exp(val - maxVal);
        sum += val;
    }
    
    for (auto& val : output) {
        val /= sum;
    }
    
    return output;
}

// Linear Model Implementation
LinearModel::LinearModel() : inputSize(0), outputSize(0) {}

LinearModel::~LinearModel() {}

void LinearModel::initialize(int inputSize, int outputSize) {
    this->inputSize = inputSize;
    this->outputSize = outputSize;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.1);
    
    weights.resize(inputSize);
    for (auto& row : weights) {
        row.resize(outputSize);
        for (auto& weight : row) {
            weight = dist(gen);
        }
    }
    
    bias.resize(outputSize);
    for (auto& b : bias) b = dist(gen);
}

std::vector<double> LinearModel::predict(const std::vector<double>& input) {
    if (input.size() != inputSize) {
        return std::vector<double>();
    }
    
    std::vector<double> output(outputSize);
    for (int j = 0; j < outputSize; ++j) {
        output[j] = bias[j];
        for (int i = 0; i < inputSize; ++i) {
            output[j] += input[i] * weights[i][j];
        }
    }
    
    return output;
}

void LinearModel::train(const std::vector<std::vector<double>>& inputs, 
                       const std::vector<std::vector<double>>& targets, 
                       double learningRate) {
    if (inputs.empty() || inputs.size() != targets.size()) {
        return;
    }
    
    int batchSize = inputs.size();
    
    for (int b = 0; b < batchSize; ++b) {
        std::vector<double> prediction = predict(inputs[b]);
        
        for (int j = 0; j < outputSize; ++j) {
            double error = targets[b][j] - prediction[j];
            bias[j] += learningRate * error;
            
            for (int i = 0; i < inputSize; ++i) {
                weights[i][j] += learningRate * error * inputs[b][i];
            }
        }
    }
}

bool LinearModel::save(const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    file.write(reinterpret_cast<const char*>(&inputSize), sizeof(inputSize));
    file.write(reinterpret_cast<const char*>(&outputSize), sizeof(outputSize));
    
    for (const auto& row : weights) {
        file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
    }
    file.write(reinterpret_cast<const char*>(bias.data()), bias.size() * sizeof(double));
    
    return true;
}

bool LinearModel::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    file.read(reinterpret_cast<char*>(&inputSize), sizeof(inputSize));
    file.read(reinterpret_cast<char*>(&outputSize), sizeof(outputSize));
    
    weights.resize(inputSize);
    for (auto& row : weights) {
        row.resize(outputSize);
        file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
    }
    
    bias.resize(outputSize);
    file.read(reinterpret_cast<char*>(bias.data()), bias.size() * sizeof(double));
    
    return true;
}

// Logistic Model Implementation
LogisticModel::LogisticModel() : inputSize(0), outputSize(0) {}

LogisticModel::~LogisticModel() {}

void LogisticModel::initialize(int inputSize, int outputSize) {
    this->inputSize = inputSize;
    this->outputSize = outputSize;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.1);
    
    weights.resize(inputSize);
    for (auto& row : weights) {
        row.resize(outputSize);
        for (auto& weight : row) {
            weight = dist(gen);
        }
    }
    
    bias.resize(outputSize);
    for (auto& b : bias) b = dist(gen);
}

std::vector<double> LogisticModel::predict(const std::vector<double>& input) {
    if (input.size() != inputSize) {
        return std::vector<double>();
    }
    
    std::vector<double> output(outputSize);
    for (int j = 0; j < outputSize; ++j) {
        output[j] = bias[j];
        for (int i = 0; i < inputSize; ++i) {
            output[j] += input[i] * weights[i][j];
        }
    }
    
    return softmax(output);
}

void LogisticModel::train(const std::vector<std::vector<double>>& inputs, 
                         const std::vector<std::vector<double>>& targets, 
                         double learningRate) {
    if (inputs.empty() || inputs.size() != targets.size()) {
        return;
    }
    
    int batchSize = inputs.size();
    
    for (int b = 0; b < batchSize; ++b) {
        std::vector<double> prediction = predict(inputs[b]);
        
        for (int j = 0; j < outputSize; ++j) {
            double error = targets[b][j] - prediction[j];
            bias[j] += learningRate * error;
            
            for (int i = 0; i < inputSize; ++i) {
                weights[i][j] += learningRate * error * inputs[b][i];
            }
        }
    }
}

bool LogisticModel::save(const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    file.write(reinterpret_cast<const char*>(&inputSize), sizeof(inputSize));
    file.write(reinterpret_cast<const char*>(&outputSize), sizeof(outputSize));
    
    for (const auto& row : weights) {
        file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
    }
    file.write(reinterpret_cast<const char*>(bias.data()), bias.size() * sizeof(double));
    
    return true;
}

bool LogisticModel::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    file.read(reinterpret_cast<char*>(&inputSize), sizeof(inputSize));
    file.read(reinterpret_cast<char*>(&outputSize), sizeof(outputSize));
    
    weights.resize(inputSize);
    for (auto& row : weights) {
        row.resize(outputSize);
        file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
    }
    
    bias.resize(outputSize);
    file.read(reinterpret_cast<char*>(bias.data()), bias.size() * sizeof(double));
    
    return true;
}

double LogisticModel::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

std::vector<double> LogisticModel::softmax(const std::vector<double>& input) {
    std::vector<double> output = input;
    double maxVal = *std::max_element(output.begin(), output.end());
    
    double sum = 0.0;
    for (auto& val : output) {
        val = std::exp(val - maxVal);
        sum += val;
    }
    
    for (auto& val : output) {
        val /= sum;
    }
    
    return output;
}

// Random Forest Implementation (simplified)
RandomForest::RandomForest() : inputSize(0), outputSize(0), numTrees(10) {}

RandomForest::~RandomForest() {
    for (auto tree : trees) {
        cleanupTree(tree);
    }
}

void RandomForest::initialize(int inputSize, int outputSize) {
    this->inputSize = inputSize;
    this->outputSize = outputSize;
    trees.clear();
}

std::vector<double> RandomForest::predict(const std::vector<double>& input) {
    if (input.size() != inputSize || trees.empty()) {
        return std::vector<double>(outputSize, 0.0);
    }
    
    std::vector<double> prediction(outputSize, 0.0);
    
    for (auto tree : trees) {
        double treePrediction = predictTree(tree, input);
        int classIndex = static_cast<int>(treePrediction) % outputSize;
        prediction[classIndex] += 1.0;
    }
    
    // Normalize
    double sum = 0.0;
    for (auto& val : prediction) sum += val;
    if (sum > 0) {
        for (auto& val : prediction) val /= sum;
    }
    
    return prediction;
}

void RandomForest::train(const std::vector<std::vector<double>>& inputs, 
                        const std::vector<std::vector<double>>& targets, 
                        double learningRate) {
    if (inputs.empty()) return;
    
    // Convert targets to labels
    std::vector<int> labels;
    for (const auto& target : targets) {
        auto maxIt = std::max_element(target.begin(), target.end());
        labels.push_back(std::distance(target.begin(), maxIt));
    }
    
    // Build trees
    for (int i = 0; i < numTrees; ++i) {
        DecisionTree* tree = buildTree(inputs, labels);
        trees.push_back(tree);
    }
}

bool RandomForest::save(const std::string& path) {
    // Simplified save - just save the number of trees
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    file.write(reinterpret_cast<const char*>(&numTrees), sizeof(numTrees));
    return true;
}

bool RandomForest::load(const std::string& path) {
    // Simplified load
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    file.read(reinterpret_cast<char*>(&numTrees), sizeof(numTrees));
    return true;
}

RandomForest::DecisionTree* RandomForest::buildTree(const std::vector<std::vector<double>>& data, 
                                                   const std::vector<int>& labels, 
                                                   int depth) {
    if (depth > 10 || data.size() < 5) {
        // Create leaf node
        DecisionTree* leaf = new DecisionTree();
        leaf->featureIndex = -1;
        
        // Calculate most common label
        std::vector<int> labelCounts(outputSize, 0);
        for (int label : labels) {
            if (label >= 0 && label < outputSize) {
                labelCounts[label]++;
            }
        }
        auto maxIt = std::max_element(labelCounts.begin(), labelCounts.end());
        leaf->value = static_cast<double>(std::distance(labelCounts.begin(), maxIt));
        
        return leaf;
    }
    
    // Find best split
    int bestFeature = -1;
    double bestThreshold = 0.0;
    double bestGain = -1.0;
    
    for (int feature = 0; feature < inputSize; ++feature) {
        for (size_t i = 0; i < data.size(); ++i) {
            double threshold = data[i][feature];
            
            // Calculate information gain
            std::vector<int> leftLabels, rightLabels;
            for (size_t j = 0; j < data.size(); ++j) {
                if (data[j][feature] <= threshold) {
                    leftLabels.push_back(labels[j]);
                } else {
                    rightLabels.push_back(labels[j]);
                }
            }
            
            if (leftLabels.size() > 0 && rightLabels.size() > 0) {
                double gain = 1.0; // Simplified gain calculation
                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeature = feature;
                    bestThreshold = threshold;
                }
            }
        }
    }
    
    if (bestFeature == -1) {
        // Create leaf node
        DecisionTree* leaf = new DecisionTree();
        leaf->featureIndex = -1;
        std::vector<int> labelCounts(outputSize, 0);
        for (int label : labels) {
            if (label >= 0 && label < outputSize) {
                labelCounts[label]++;
            }
        }
        auto maxIt = std::max_element(labelCounts.begin(), labelCounts.end());
        leaf->value = static_cast<double>(std::distance(labelCounts.begin(), maxIt));
        return leaf;
    }
    
    // Split data
    std::vector<std::vector<double>> leftData, rightData;
    std::vector<int> leftLabels, rightLabels;
    
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i][bestFeature] <= bestThreshold) {
            leftData.push_back(data[i]);
            leftLabels.push_back(labels[i]);
        } else {
            rightData.push_back(data[i]);
            rightLabels.push_back(labels[i]);
        }
    }
    
    // Create node
    DecisionTree* node = new DecisionTree();
    node->featureIndex = bestFeature;
    node->threshold = bestThreshold;
    node->left = buildTree(leftData, leftLabels, depth + 1);
    node->right = buildTree(rightData, rightLabels, depth + 1);
    
    return node;
}

double RandomForest::predictTree(DecisionTree* tree, const std::vector<double>& input) {
    if (!tree) return 0.0;
    
    if (tree->featureIndex == -1) {
        return tree->value;
    }
    
    if (input[tree->featureIndex] <= tree->threshold) {
        return predictTree(tree->left, input);
    } else {
        return predictTree(tree->right, input);
    }
}

void RandomForest::cleanupTree(DecisionTree* tree) {
    if (!tree) return;
    
    cleanupTree(tree->left);
    cleanupTree(tree->right);
    delete tree;
}

// ModelTrainer Implementation
ModelTrainer::ModelTrainer() : trainingFlag(false), pauseFlag(false), stopFlag(false) {
    rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());
    dist = std::uniform_real_distribution<double>(-0.1, 0.1);
}

ModelTrainer::~ModelTrainer() {
    stopTraining();
    if (trainingThread.joinable()) {
        trainingThread.join();
    }
}

void ModelTrainer::setDatasetPath(const std::string& path) {
    config.datasetPath = path;
}

void ModelTrainer::setOutputPath(const std::string& path) {
    config.outputPath = path;
}

void ModelTrainer::setModelType(const std::string& type) {
    config.modelType = type;
}

void ModelTrainer::setInputSize(int size) {
    config.inputSize = size;
}

void ModelTrainer::setHiddenSize(int size) {
    config.hiddenSize = size;
}

void ModelTrainer::setOutputSize(int size) {
    config.outputSize = size;
}

void ModelTrainer::setLearningRate(double rate) {
    config.learningRate = rate;
}

void ModelTrainer::setBatchSize(int size) {
    config.batchSize = size;
}

void ModelTrainer::setEpochs(int epochs) {
    config.epochs = epochs;
}

void ModelTrainer::setProgressCallback(std::function<void(int, double, double)> callback) {
    progressCallback = callback;
}

void ModelTrainer::startTraining() {
    std::lock_guard<std::mutex> lock(trainingMutex);
    
    if (trainingFlag) {
        return; // Already training
    }
    
    trainingFlag = true;
    pauseFlag = false;
    stopFlag = false;
    
    // Reset metrics
    metrics = TrainingMetrics();
    
    // Start training in a separate thread
    trainingThread = std::thread([this]() {
        createModel();
        
        if (config.modelType == "neural") {
            trainNeuralNetwork();
        } else if (config.modelType == "linear") {
            trainLinearModel();
        } else if (config.modelType == "logistic") {
            trainLogisticModel();
        } else if (config.modelType == "randomforest") {
            trainRandomForest();
        }
        
        trainingFlag = false;
    });
}

void ModelTrainer::stopTraining() {
    std::lock_guard<std::mutex> lock(trainingMutex);
    stopFlag = true;
    trainingFlag = false;
}

void ModelTrainer::pauseTraining() {
    pauseFlag = true;
}

void ModelTrainer::resumeTraining() {
    pauseFlag = false;
}

bool ModelTrainer::saveModel(const std::string& path) {
    if (config.modelType == "neural" && neuralNetwork) {
        return neuralNetwork->save(path);
    } else if (config.modelType == "linear" && linearModel) {
        return linearModel->save(path);
    } else if (config.modelType == "logistic" && logisticModel) {
        return logisticModel->save(path);
    } else if (config.modelType == "randomforest" && randomForest) {
        return randomForest->save(path);
    }
    return false;
}

bool ModelTrainer::loadModel(const std::string& path) {
    createModel();
    
    if (config.modelType == "neural" && neuralNetwork) {
        return neuralNetwork->load(path);
    } else if (config.modelType == "linear" && linearModel) {
        return linearModel->load(path);
    } else if (config.modelType == "logistic" && logisticModel) {
        return logisticModel->load(path);
    } else if (config.modelType == "randomforest" && randomForest) {
        return randomForest->load(path);
    }
    return false;
}

TrainingConfig ModelTrainer::getConfig() const {
    return config;
}

TrainingMetrics ModelTrainer::getMetrics() const {
    return metrics;
}

bool ModelTrainer::isTraining() const {
    return trainingFlag;
}

bool ModelTrainer::isPaused() const {
    return pauseFlag;
}

void ModelTrainer::createModel() {
    if (config.modelType == "neural") {
        neuralNetwork = std::make_unique<NeuralNetwork>();
        neuralNetwork->setHiddenSize(config.hiddenSize);
        neuralNetwork->initialize(config.inputSize, config.outputSize);
    } else if (config.modelType == "linear") {
        linearModel = std::make_unique<LinearModel>();
        linearModel->initialize(config.inputSize, config.outputSize);
    } else if (config.modelType == "logistic") {
        logisticModel = std::make_unique<LogisticModel>();
        logisticModel->initialize(config.inputSize, config.outputSize);
    } else if (config.modelType == "randomforest") {
        randomForest = std::make_unique<RandomForest>();
        randomForest->initialize(config.inputSize, config.outputSize);
    }
}

void ModelTrainer::initializeModel() {
    // Model initialization is done in createModel()
}

void ModelTrainer::cleanupModel() {
    neuralNetwork.reset();
    linearModel.reset();
    logisticModel.reset();
    randomForest.reset();
}

void ModelTrainer::trainNeuralNetwork() {
    if (!neuralNetwork) return;
    
    // Generate dummy training data
    generateDummyData();
    
    for (int epoch = 0; epoch < config.epochs && !stopFlag; ++epoch) {
        while (pauseFlag && !stopFlag) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        if (stopFlag) break;
        
        // Training loop
        double totalLoss = 0.0;
        int correctPredictions = 0;
        int totalPredictions = 0;
        
        for (size_t i = 0; i < trainingData.size(); i += config.batchSize) {
            if (stopFlag) break;
            
            std::vector<std::vector<double>> batch;
            std::vector<std::vector<double>> targets;
            
            for (int j = 0; j < config.batchSize && i + j < trainingData.size(); ++j) {
                batch.push_back(trainingData[i + j]);
                
                // Create one-hot target
                std::vector<double> target(config.outputSize, 0.0);
                target[trainingLabels[i + j]] = 1.0;
                targets.push_back(target);
            }
            
            if (!batch.empty()) {
                neuralNetwork->train(batch, targets, config.learningRate);
                
                // Calculate metrics
                for (size_t k = 0; k < batch.size(); ++k) {
                    auto prediction = neuralNetwork->predict(batch[k]);
                    totalLoss += calculateLoss(prediction, targets[k]);
                    
                    auto maxIt = std::max_element(prediction.begin(), prediction.end());
                    int predictedClass = std::distance(prediction.begin(), maxIt);
                    if (predictedClass == trainingLabels[i + k]) {
                        correctPredictions++;
                    }
                    totalPredictions++;
                }
            }
        }
        
        // Update metrics
        metrics.currentEpoch = epoch + 1;
        metrics.loss = totalLoss / totalPredictions;
        metrics.accuracy = static_cast<double>(correctPredictions) / totalPredictions;
        metrics.lossHistory.push_back(metrics.loss);
        metrics.accuracyHistory.push_back(metrics.accuracy);
        
        // Call progress callback
        if (progressCallback) {
            progressCallback(epoch + 1, metrics.loss, metrics.accuracy);
        }
        
        // Small delay to prevent overwhelming the console
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void ModelTrainer::trainLinearModel() {
    if (!linearModel) return;
    
    generateDummyData();
    
    for (int epoch = 0; epoch < config.epochs && !stopFlag; ++epoch) {
        while (pauseFlag && !stopFlag) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        if (stopFlag) break;
        
        double totalLoss = 0.0;
        int correctPredictions = 0;
        int totalPredictions = 0;
        
        for (size_t i = 0; i < trainingData.size(); i += config.batchSize) {
            if (stopFlag) break;
            
            std::vector<std::vector<double>> batch;
            std::vector<std::vector<double>> targets;
            
            for (int j = 0; j < config.batchSize && i + j < trainingData.size(); ++j) {
                batch.push_back(trainingData[i + j]);
                
                std::vector<double> target(config.outputSize, 0.0);
                target[trainingLabels[i + j]] = 1.0;
                targets.push_back(target);
            }
            
            if (!batch.empty()) {
                linearModel->train(batch, targets, config.learningRate);
                
                for (size_t k = 0; k < batch.size(); ++k) {
                    auto prediction = linearModel->predict(batch[k]);
                    totalLoss += calculateLoss(prediction, targets[k]);
                    
                    auto maxIt = std::max_element(prediction.begin(), prediction.end());
                    int predictedClass = std::distance(prediction.begin(), maxIt);
                    if (predictedClass == trainingLabels[i + k]) {
                        correctPredictions++;
                    }
                    totalPredictions++;
                }
            }
        }
        
        metrics.currentEpoch = epoch + 1;
        metrics.loss = totalLoss / totalPredictions;
        metrics.accuracy = static_cast<double>(correctPredictions) / totalPredictions;
        metrics.lossHistory.push_back(metrics.loss);
        metrics.accuracyHistory.push_back(metrics.accuracy);
        
        if (progressCallback) {
            progressCallback(epoch + 1, metrics.loss, metrics.accuracy);
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void ModelTrainer::trainLogisticModel() {
    if (!logisticModel) return;
    
    generateDummyData();
    
    for (int epoch = 0; epoch < config.epochs && !stopFlag; ++epoch) {
        while (pauseFlag && !stopFlag) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        if (stopFlag) break;
        
        double totalLoss = 0.0;
        int correctPredictions = 0;
        int totalPredictions = 0;
        
        for (size_t i = 0; i < trainingData.size(); i += config.batchSize) {
            if (stopFlag) break;
            
            std::vector<std::vector<double>> batch;
            std::vector<std::vector<double>> targets;
            
            for (int j = 0; j < config.batchSize && i + j < trainingData.size(); ++j) {
                batch.push_back(trainingData[i + j]);
                
                std::vector<double> target(config.outputSize, 0.0);
                target[trainingLabels[i + j]] = 1.0;
                targets.push_back(target);
            }
            
            if (!batch.empty()) {
                logisticModel->train(batch, targets, config.learningRate);
                
                for (size_t k = 0; k < batch.size(); ++k) {
                    auto prediction = logisticModel->predict(batch[k]);
                    totalLoss += calculateLoss(prediction, targets[k]);
                    
                    auto maxIt = std::max_element(prediction.begin(), prediction.end());
                    int predictedClass = std::distance(prediction.begin(), maxIt);
                    if (predictedClass == trainingLabels[i + k]) {
                        correctPredictions++;
                    }
                    totalPredictions++;
                }
            }
        }
        
        metrics.currentEpoch = epoch + 1;
        metrics.loss = totalLoss / totalPredictions;
        metrics.accuracy = static_cast<double>(correctPredictions) / totalPredictions;
        metrics.lossHistory.push_back(metrics.loss);
        metrics.accuracyHistory.push_back(metrics.accuracy);
        
        if (progressCallback) {
            progressCallback(epoch + 1, metrics.loss, metrics.accuracy);
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void ModelTrainer::trainRandomForest() {
    if (!randomForest) return;
    
    generateDummyData();
    
    // Convert data for random forest
    std::vector<std::vector<double>> targets;
    for (int label : trainingLabels) {
        std::vector<double> target(config.outputSize, 0.0);
        target[label] = 1.0;
        targets.push_back(target);
    }
    
    randomForest->train(trainingData, targets, config.learningRate);
    
    // Calculate final metrics
    double totalLoss = 0.0;
    int correctPredictions = 0;
    int totalPredictions = 0;
    
    for (size_t i = 0; i < trainingData.size(); ++i) {
        auto prediction = randomForest->predict(trainingData[i]);
        totalLoss += calculateLoss(prediction, targets[i]);
        
        auto maxIt = std::max_element(prediction.begin(), prediction.end());
        int predictedClass = std::distance(prediction.begin(), maxIt);
        if (predictedClass == trainingLabels[i]) {
            correctPredictions++;
        }
        totalPredictions++;
    }
    
    metrics.currentEpoch = config.epochs;
    metrics.loss = totalLoss / totalPredictions;
    metrics.accuracy = static_cast<double>(correctPredictions) / totalPredictions;
    metrics.lossHistory.push_back(metrics.loss);
    metrics.accuracyHistory.push_back(metrics.accuracy);
    
    if (progressCallback) {
        progressCallback(config.epochs, metrics.loss, metrics.accuracy);
    }
}

double ModelTrainer::calculateLoss(const std::vector<double>& predictions, const std::vector<double>& targets) {
    if (predictions.size() != targets.size()) return 0.0;
    
    double loss = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        double diff = predictions[i] - targets[i];
        loss += diff * diff; // Mean squared error
    }
    return loss / predictions.size();
}

double ModelTrainer::calculateAccuracy(const std::vector<double>& predictions, const std::vector<int>& targets) {
    if (predictions.size() != targets.size()) return 0.0;
    
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (static_cast<int>(predictions[i] + 0.5) == targets[i]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / predictions.size();
}

std::vector<double> ModelTrainer::forwardPass(const std::vector<double>& input) {
    if (config.modelType == "neural" && neuralNetwork) {
        return neuralNetwork->predict(input);
    } else if (config.modelType == "linear" && linearModel) {
        return linearModel->predict(input);
    } else if (config.modelType == "logistic" && logisticModel) {
        return logisticModel->predict(input);
    } else if (config.modelType == "randomforest" && randomForest) {
        return randomForest->predict(input);
    }
    return std::vector<double>();
}

void ModelTrainer::backwardPass(const std::vector<double>& input, const std::vector<double>& target) {
    // This would implement backpropagation for neural networks
    // For now, it's handled in the individual model train methods
}

std::vector<std::vector<double>> ModelTrainer::getBatch(int batchIndex) {
    std::vector<std::vector<double>> batch;
    int start = batchIndex * config.batchSize;
    int end = std::min(start + config.batchSize, static_cast<int>(trainingData.size()));
    
    for (int i = start; i < end; ++i) {
        batch.push_back(trainingData[i]);
    }
    
    return batch;
}

std::vector<int> ModelTrainer::getBatchLabels(int batchIndex) {
    std::vector<int> labels;
    int start = batchIndex * config.batchSize;
    int end = std::min(start + config.batchSize, static_cast<int>(trainingLabels.size()));
    
    for (int i = start; i < end; ++i) {
        labels.push_back(trainingLabels[i]);
    }
    
    return labels;
}

void ModelTrainer::normalizeData() {
    // Simple normalization - scale to [0, 1]
    if (trainingData.empty()) return;
    
    int numFeatures = trainingData[0].size();
    std::vector<double> minVals(numFeatures, std::numeric_limits<double>::max());
    std::vector<double> maxVals(numFeatures, std::numeric_limits<double>::lowest());
    
    // Find min and max values
    for (const auto& sample : trainingData) {
        for (size_t i = 0; i < sample.size() && i < numFeatures; ++i) {
            minVals[i] = std::min(minVals[i], sample[i]);
            maxVals[i] = std::max(maxVals[i], sample[i]);
        }
    }
    
    // Normalize
    for (auto& sample : trainingData) {
        for (size_t i = 0; i < sample.size() && i < numFeatures; ++i) {
            if (maxVals[i] > minVals[i]) {
                sample[i] = (sample[i] - minVals[i]) / (maxVals[i] - minVals[i]);
            }
        }
    }
}

void ModelTrainer::shuffleData() {
    // Fisher-Yates shuffle
    for (size_t i = trainingData.size() - 1; i > 0; --i) {
        size_t j = dist(rng) * (i + 1);
        std::swap(trainingData[i], trainingData[j]);
        std::swap(trainingLabels[i], trainingLabels[j]);
    }
}

void ModelTrainer::generateDummyData() {
    // Generate synthetic training data
    trainingData.clear();
    trainingLabels.clear();
    
    int numSamples = 1000; // Generate 1000 samples
    
    for (int i = 0; i < numSamples; ++i) {
        std::vector<double> sample(config.inputSize);
        
        // Generate features
        for (int j = 0; j < config.inputSize; ++j) {
            sample[j] = dist(rng);
        }
        
        // Generate label based on features (simple rule)
        int label = 0;
        if (sample[0] > 0.5) label = 1;
        if (sample[1] > 0.7) label = 2;
        label = label % config.outputSize;
        
        trainingData.push_back(sample);
        trainingLabels.push_back(label);
    }
    
    normalizeData();
} 