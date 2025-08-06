#include "modeltrainer.h"
#include <QFile>
#include <QTextStream>
#include <QDebug>
#include <QDir>
#include <QDateTime>
#include <algorithm>
#include <cmath>
#include <random>

// ModelTrainer implementation
ModelTrainer::ModelTrainer(QObject *parent)
    : QObject(parent)
    , trainingFlag(false)
    , pauseFlag(false)
    , stopFlag(false)
    , rng(std::random_device{}())
    , dist(0.0, 1.0)
{
    trainingTimer = new QTimer(this);
    trainingTimer->setSingleShot(true);
    connect(trainingTimer, &QTimer::timeout, this, &ModelTrainer::trainingLoop);
}

ModelTrainer::~ModelTrainer()
{
    stopTraining();
    cleanupModel();
}

void ModelTrainer::setDatasetPath(const QString &path)
{
    config.datasetPath = path;
}

void ModelTrainer::setOutputPath(const QString &path)
{
    config.outputPath = path;
}

void ModelTrainer::setModelType(const QString &type)
{
    config.modelType = type;
}

void ModelTrainer::setInputSize(int size)
{
    config.inputSize = size;
}

void ModelTrainer::setHiddenSize(int size)
{
    config.hiddenSize = size;
}

void ModelTrainer::setOutputSize(int size)
{
    config.outputSize = size;
}

void ModelTrainer::setLearningRate(double rate)
{
    config.learningRate = rate;
}

void ModelTrainer::setBatchSize(int size)
{
    config.batchSize = size;
}

void ModelTrainer::setEpochs(int epochs)
{
    config.epochs = epochs;
}

void ModelTrainer::startTraining()
{
    QMutexLocker locker(&trainingMutex);
    
    if (trainingFlag) {
        return;
    }
    
    // Reset flags
    trainingFlag = true;
    pauseFlag = false;
    stopFlag = false;
    
    // Reset metrics
    metrics = TrainingMetrics();
    
    // Create and initialize model
    createModel();
    initializeModel();
    
    // Load and prepare data
    if (!loadTrainingData()) {
        emit trainingError("Failed to load training data");
        trainingFlag = false;
        return;
    }
    
    // Start training loop
    trainingTimer->start(10); // Start immediately
}

void ModelTrainer::stopTraining()
{
    QMutexLocker locker(&trainingMutex);
    stopFlag = true;
    trainingFlag = false;
    pauseFlag = false;
    trainingTimer->stop();
}

void ModelTrainer::pauseTraining()
{
    QMutexLocker locker(&trainingMutex);
    pauseFlag = true;
}

void ModelTrainer::resumeTraining()
{
    QMutexLocker locker(&trainingMutex);
    pauseFlag = false;
    if (trainingFlag && !stopFlag) {
        trainingTimer->start(10);
    }
}

bool ModelTrainer::saveModel(const QString &path)
{
    QMutexLocker locker(&trainingMutex);
    
    if (config.modelType == "Neural Network" && neuralNetwork) {
        return neuralNetwork->save(path);
    } else if (config.modelType == "Linear Regression" && linearModel) {
        return linearModel->save(path);
    } else if (config.modelType == "Logistic Regression" && logisticModel) {
        return logisticModel->save(path);
    } else if (config.modelType == "Random Forest" && randomForest) {
        return randomForest->save(path);
    }
    
    return false;
}

bool ModelTrainer::loadModel(const QString &path)
{
    QMutexLocker locker(&trainingMutex);
    
    createModel();
    
    if (config.modelType == "Neural Network" && neuralNetwork) {
        return neuralNetwork->load(path);
    } else if (config.modelType == "Linear Regression" && linearModel) {
        return linearModel->load(path);
    } else if (config.modelType == "Logistic Regression" && logisticModel) {
        return logisticModel->load(path);
    } else if (config.modelType == "Random Forest" && randomForest) {
        return randomForest->load(path);
    }
    
    return false;
}

TrainingConfig ModelTrainer::getConfig() const
{
    return config;
}

TrainingMetrics ModelTrainer::getMetrics() const
{
    return metrics;
}

bool ModelTrainer::isTraining() const
{
    return trainingFlag;
}

bool ModelTrainer::isPaused() const
{
    return pauseFlag;
}

void ModelTrainer::trainingLoop()
{
    if (stopFlag || !trainingFlag) {
        trainingFlag = false;
        emit trainingFinished();
        return;
    }
    
    if (pauseFlag) {
        trainingTimer->start(100); // Check again in 100ms
        return;
    }
    
    // Perform one epoch of training
    if (metrics.currentEpoch < config.epochs) {
        metrics.currentEpoch++;
        
        // Train the model based on type
        if (config.modelType == "Neural Network") {
            trainNeuralNetwork();
        } else if (config.modelType == "Linear Regression") {
            trainLinearModel();
        } else if (config.modelType == "Logistic Regression") {
            trainLogisticModel();
        } else if (config.modelType == "Random Forest") {
            trainRandomForest();
        }
        
        // Update metrics
        metrics.lossHistory.append(metrics.loss);
        metrics.accuracyHistory.append(metrics.accuracy);
        
        // Emit progress signal
        emit progressUpdated(metrics.currentEpoch, metrics.loss, metrics.accuracy);
        
        // Continue training
        trainingTimer->start(10);
    } else {
        // Training complete
        trainingFlag = false;
        emit trainingFinished();
    }
}

void ModelTrainer::createModel()
{
    cleanupModel();
    
    if (config.modelType == "Neural Network") {
        neuralNetwork = std::make_unique<NeuralNetwork>();
    } else if (config.modelType == "Linear Regression") {
        linearModel = std::make_unique<LinearModel>();
    } else if (config.modelType == "Logistic Regression") {
        logisticModel = std::make_unique<LogisticModel>();
    } else if (config.modelType == "Random Forest") {
        randomForest = std::make_unique<RandomForest>();
    }
}

void ModelTrainer::initializeModel()
{
    if (config.modelType == "Neural Network" && neuralNetwork) {
        neuralNetwork->setHiddenSize(config.hiddenSize);
        neuralNetwork->initialize(config.inputSize, config.outputSize);
    } else if (config.modelType == "Linear Regression" && linearModel) {
        linearModel->initialize(config.inputSize, config.outputSize);
    } else if (config.modelType == "Logistic Regression" && logisticModel) {
        logisticModel->initialize(config.inputSize, config.outputSize);
    } else if (config.modelType == "Random Forest" && randomForest) {
        randomForest->initialize(config.inputSize, config.outputSize);
    }
}

void ModelTrainer::cleanupModel()
{
    neuralNetwork.reset();
    linearModel.reset();
    logisticModel.reset();
    randomForest.reset();
}

bool ModelTrainer::loadTrainingData()
{
    // For demonstration, create synthetic data
    // In a real implementation, this would load from the dataset path
    
    int numSamples = 1000;
    trainingData.resize(numSamples);
    trainingLabels.resize(numSamples);
    
    for (int i = 0; i < numSamples; ++i) {
        trainingData[i].resize(config.inputSize);
        for (int j = 0; j < config.inputSize; ++j) {
            trainingData[i][j] = dist(rng);
        }
        trainingLabels[i] = i % config.outputSize;
    }
    
    // Create validation data
    int numValidation = numSamples / 5;
    validationData.resize(numValidation);
    validationLabels.resize(numValidation);
    
    for (int i = 0; i < numValidation; ++i) {
        validationData[i].resize(config.inputSize);
        for (int j = 0; j < config.inputSize; ++j) {
            validationData[i][j] = dist(rng);
        }
        validationLabels[i] = i % config.outputSize;
    }
    
    return true;
}

void ModelTrainer::trainNeuralNetwork()
{
    if (!neuralNetwork) return;
    
    // Convert labels to one-hot encoding
    QVector<QVector<double>> targets;
    targets.resize(trainingData.size());
    for (int i = 0; i < trainingData.size(); ++i) {
        targets[i].resize(config.outputSize, 0.0);
        targets[i][trainingLabels[i]] = 1.0;
    }
    
    // Train the model
    neuralNetwork->train(trainingData, targets, config.learningRate);
    
    // Calculate metrics
    double totalLoss = 0.0;
    int correctPredictions = 0;
    
    for (int i = 0; i < trainingData.size(); ++i) {
        QVector<double> prediction = neuralNetwork->predict(trainingData[i]);
        totalLoss += calculateLoss(prediction, targets[i]);
        
        // Find predicted class
        int predictedClass = 0;
        double maxProb = prediction[0];
        for (int j = 1; j < prediction.size(); ++j) {
            if (prediction[j] > maxProb) {
                maxProb = prediction[j];
                predictedClass = j;
            }
        }
        
        if (predictedClass == trainingLabels[i]) {
            correctPredictions++;
        }
    }
    
    metrics.loss = totalLoss / trainingData.size();
    metrics.accuracy = static_cast<double>(correctPredictions) / trainingData.size();
}

void ModelTrainer::trainLinearModel()
{
    if (!linearModel) return;
    
    // Convert labels to one-hot encoding
    QVector<QVector<double>> targets;
    targets.resize(trainingData.size());
    for (int i = 0; i < trainingData.size(); ++i) {
        targets[i].resize(config.outputSize, 0.0);
        targets[i][trainingLabels[i]] = 1.0;
    }
    
    // Train the model
    linearModel->train(trainingData, targets, config.learningRate);
    
    // Calculate metrics
    double totalLoss = 0.0;
    int correctPredictions = 0;
    
    for (int i = 0; i < trainingData.size(); ++i) {
        QVector<double> prediction = linearModel->predict(trainingData[i]);
        totalLoss += calculateLoss(prediction, targets[i]);
        
        // Find predicted class
        int predictedClass = 0;
        double maxProb = prediction[0];
        for (int j = 1; j < prediction.size(); ++j) {
            if (prediction[j] > maxProb) {
                maxProb = prediction[j];
                predictedClass = j;
            }
        }
        
        if (predictedClass == trainingLabels[i]) {
            correctPredictions++;
        }
    }
    
    metrics.loss = totalLoss / trainingData.size();
    metrics.accuracy = static_cast<double>(correctPredictions) / trainingData.size();
}

void ModelTrainer::trainLogisticModel()
{
    if (!logisticModel) return;
    
    // Convert labels to one-hot encoding
    QVector<QVector<double>> targets;
    targets.resize(trainingData.size());
    for (int i = 0; i < trainingData.size(); ++i) {
        targets[i].resize(config.outputSize, 0.0);
        targets[i][trainingLabels[i]] = 1.0;
    }
    
    // Train the model
    logisticModel->train(trainingData, targets, config.learningRate);
    
    // Calculate metrics
    double totalLoss = 0.0;
    int correctPredictions = 0;
    
    for (int i = 0; i < trainingData.size(); ++i) {
        QVector<double> prediction = logisticModel->predict(trainingData[i]);
        totalLoss += calculateLoss(prediction, targets[i]);
        
        // Find predicted class
        int predictedClass = 0;
        double maxProb = prediction[0];
        for (int j = 1; j < prediction.size(); ++j) {
            if (prediction[j] > maxProb) {
                maxProb = prediction[j];
                predictedClass = j;
            }
        }
        
        if (predictedClass == trainingLabels[i]) {
            correctPredictions++;
        }
    }
    
    metrics.loss = totalLoss / trainingData.size();
    metrics.accuracy = static_cast<double>(correctPredictions) / trainingData.size();
}

void ModelTrainer::trainRandomForest()
{
    if (!randomForest) return;
    
    // Convert labels to one-hot encoding
    QVector<QVector<double>> targets;
    targets.resize(trainingData.size());
    for (int i = 0; i < trainingData.size(); ++i) {
        targets[i].resize(config.outputSize, 0.0);
        targets[i][trainingLabels[i]] = 1.0;
    }
    
    // Train the model
    randomForest->train(trainingData, targets, config.learningRate);
    
    // Calculate metrics
    double totalLoss = 0.0;
    int correctPredictions = 0;
    
    for (int i = 0; i < trainingData.size(); ++i) {
        QVector<double> prediction = randomForest->predict(trainingData[i]);
        totalLoss += calculateLoss(prediction, targets[i]);
        
        // Find predicted class
        int predictedClass = 0;
        double maxProb = prediction[0];
        for (int j = 1; j < prediction.size(); ++j) {
            if (prediction[j] > maxProb) {
                maxProb = prediction[j];
                predictedClass = j;
            }
        }
        
        if (predictedClass == trainingLabels[i]) {
            correctPredictions++;
        }
    }
    
    metrics.loss = totalLoss / trainingData.size();
    metrics.accuracy = static_cast<double>(correctPredictions) / trainingData.size();
}

double ModelTrainer::calculateLoss(const QVector<double> &predictions, const QVector<double> &targets)
{
    // Cross-entropy loss
    double loss = 0.0;
    for (int i = 0; i < predictions.size(); ++i) {
        if (targets[i] > 0.0) {
            loss -= targets[i] * std::log(std::max(predictions[i], 1e-15));
        }
    }
    return loss;
}

double ModelTrainer::calculateAccuracy(const QVector<double> &predictions, const QVector<int> &targets)
{
    int correct = 0;
    for (int i = 0; i < predictions.size(); ++i) {
        if (static_cast<int>(predictions[i] + 0.5) == targets[i]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / predictions.size();
}

// Neural Network implementation
NeuralNetwork::NeuralNetwork()
    : inputSize(0), hiddenSize(128), outputSize(0)
{
}

NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork::initialize(int inputSize, int outputSize)
{
    this->inputSize = inputSize;
    this->outputSize = outputSize;
    
    // Initialize weights with Xavier initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist1(0.0, std::sqrt(2.0 / inputSize));
    std::normal_distribution<double> dist2(0.0, std::sqrt(2.0 / hiddenSize));
    
    // Initialize weights1 (input to hidden)
    weights1.resize(hiddenSize);
    for (int i = 0; i < hiddenSize; ++i) {
        weights1[i].resize(inputSize);
        for (int j = 0; j < inputSize; ++j) {
            weights1[i][j] = dist1(gen);
        }
    }
    
    // Initialize weights2 (hidden to output)
    weights2.resize(outputSize);
    for (int i = 0; i < outputSize; ++i) {
        weights2[i].resize(hiddenSize);
        for (int j = 0; j < hiddenSize; ++j) {
            weights2[i][j] = dist2(gen);
        }
    }
    
    // Initialize biases
    bias1.resize(hiddenSize, 0.0);
    bias2.resize(outputSize, 0.0);
}

QVector<double> NeuralNetwork::predict(const QVector<double> &input)
{
    // Forward pass through hidden layer
    QVector<double> hidden(hiddenSize);
    for (int i = 0; i < hiddenSize; ++i) {
        hidden[i] = bias1[i];
        for (int j = 0; j < inputSize; ++j) {
            hidden[i] += weights1[i][j] * input[j];
        }
        hidden[i] = sigmoid(hidden[i]);
    }
    
    // Forward pass through output layer
    QVector<double> output(outputSize);
    for (int i = 0; i < outputSize; ++i) {
        output[i] = bias2[i];
        for (int j = 0; j < hiddenSize; ++j) {
            output[i] += weights2[i][j] * hidden[j];
        }
    }
    
    return softmax(output);
}

void NeuralNetwork::train(const QVector<QVector<double>> &inputs, 
                         const QVector<QVector<double>> &targets, 
                         double learningRate)
{
    // Simple gradient descent (in practice, you'd use more sophisticated optimizers)
    for (int sample = 0; sample < inputs.size(); ++sample) {
        const QVector<double> &input = inputs[sample];
        const QVector<double> &target = targets[sample];
        
        // Forward pass
        QVector<double> hidden(hiddenSize);
        for (int i = 0; i < hiddenSize; ++i) {
            hidden[i] = bias1[i];
            for (int j = 0; j < inputSize; ++j) {
                hidden[i] += weights1[i][j] * input[j];
            }
            hidden[i] = sigmoid(hidden[i]);
        }
        
        QVector<double> output(outputSize);
        for (int i = 0; i < outputSize; ++i) {
            output[i] = bias2[i];
            for (int j = 0; j < hiddenSize; ++j) {
                output[i] += weights2[i][j] * hidden[j];
            }
        }
        output = softmax(output);
        
        // Backward pass (simplified)
        // In practice, you'd implement proper backpropagation
        // This is a simplified version for demonstration
        for (int i = 0; i < outputSize; ++i) {
            double error = output[i] - target[i];
            bias2[i] -= learningRate * error;
            
            for (int j = 0; j < hiddenSize; ++j) {
                weights2[i][j] -= learningRate * error * hidden[j];
            }
        }
    }
}

bool NeuralNetwork::save(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return false;
    }
    
    QTextStream out(&file);
    out << "NeuralNetwork\n";
    out << inputSize << " " << hiddenSize << " " << outputSize << "\n";
    
    // Save weights and biases
    // Implementation would save the actual weights and biases
    
    file.close();
    return true;
}

bool NeuralNetwork::load(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return false;
    }
    
    QTextStream in(&file);
    QString modelType = in.readLine();
    if (modelType != "NeuralNetwork") {
        file.close();
        return false;
    }
    
    in >> inputSize >> hiddenSize >> outputSize;
    
    // Load weights and biases
    // Implementation would load the actual weights and biases
    
    file.close();
    return true;
}

void NeuralNetwork::setHiddenSize(int size)
{
    hiddenSize = size;
}

double NeuralNetwork::sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

double NeuralNetwork::sigmoidDerivative(double x)
{
    double s = sigmoid(x);
    return s * (1.0 - s);
}

QVector<double> NeuralNetwork::softmax(const QVector<double> &input)
{
    QVector<double> output(input.size());
    double maxVal = *std::max_element(input.begin(), input.end());
    double sum = 0.0;
    
    for (int i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - maxVal);
        sum += output[i];
    }
    
    for (int i = 0; i < output.size(); ++i) {
        output[i] /= sum;
    }
    
    return output;
}

// Linear Model implementation
LinearModel::LinearModel()
    : inputSize(0), outputSize(0)
{
}

LinearModel::~LinearModel()
{
}

void LinearModel::initialize(int inputSize, int outputSize)
{
    this->inputSize = inputSize;
    this->outputSize = outputSize;
    
    // Initialize weights and biases
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.01);
    
    weights.resize(outputSize);
    for (int i = 0; i < outputSize; ++i) {
        weights[i].resize(inputSize);
        for (int j = 0; j < inputSize; ++j) {
            weights[i][j] = dist(gen);
        }
    }
    
    bias.resize(outputSize, 0.0);
}

QVector<double> LinearModel::predict(const QVector<double> &input)
{
    QVector<double> output(outputSize);
    for (int i = 0; i < outputSize; ++i) {
        output[i] = bias[i];
        for (int j = 0; j < inputSize; ++j) {
            output[i] += weights[i][j] * input[j];
        }
    }
    return output;
}

void LinearModel::train(const QVector<QVector<double>> &inputs, 
                       const QVector<QVector<double>> &targets, 
                       double learningRate)
{
    // Simple gradient descent
    for (int sample = 0; sample < inputs.size(); ++sample) {
        const QVector<double> &input = inputs[sample];
        const QVector<double> &target = targets[sample];
        
        QVector<double> prediction = predict(input);
        
        // Update weights and biases
        for (int i = 0; i < outputSize; ++i) {
            double error = prediction[i] - target[i];
            bias[i] -= learningRate * error;
            
            for (int j = 0; j < inputSize; ++j) {
                weights[i][j] -= learningRate * error * input[j];
            }
        }
    }
}

bool LinearModel::save(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return false;
    }
    
    QTextStream out(&file);
    out << "LinearModel\n";
    out << inputSize << " " << outputSize << "\n";
    
    file.close();
    return true;
}

bool LinearModel::load(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return false;
    }
    
    QTextStream in(&file);
    QString modelType = in.readLine();
    if (modelType != "LinearModel") {
        file.close();
        return false;
    }
    
    in >> inputSize >> outputSize;
    
    file.close();
    return true;
}

// Logistic Model implementation
LogisticModel::LogisticModel()
    : inputSize(0), outputSize(0)
{
}

LogisticModel::~LogisticModel()
{
}

void LogisticModel::initialize(int inputSize, int outputSize)
{
    this->inputSize = inputSize;
    this->outputSize = outputSize;
    
    // Initialize weights and biases
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.01);
    
    weights.resize(outputSize);
    for (int i = 0; i < outputSize; ++i) {
        weights[i].resize(inputSize);
        for (int j = 0; j < inputSize; ++j) {
            weights[i][j] = dist(gen);
        }
    }
    
    bias.resize(outputSize, 0.0);
}

QVector<double> LogisticModel::predict(const QVector<double> &input)
{
    QVector<double> output(outputSize);
    for (int i = 0; i < outputSize; ++i) {
        output[i] = bias[i];
        for (int j = 0; j < inputSize; ++j) {
            output[i] += weights[i][j] * input[j];
        }
    }
    return softmax(output);
}

void LogisticModel::train(const QVector<QVector<double>> &inputs, 
                         const QVector<QVector<double>> &targets, 
                         double learningRate)
{
    // Gradient descent for logistic regression
    for (int sample = 0; sample < inputs.size(); ++sample) {
        const QVector<double> &input = inputs[sample];
        const QVector<double> &target = targets[sample];
        
        QVector<double> prediction = predict(input);
        
        // Update weights and biases
        for (int i = 0; i < outputSize; ++i) {
            double error = prediction[i] - target[i];
            bias[i] -= learningRate * error;
            
            for (int j = 0; j < inputSize; ++j) {
                weights[i][j] -= learningRate * error * input[j];
            }
        }
    }
}

bool LogisticModel::save(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return false;
    }
    
    QTextStream out(&file);
    out << "LogisticModel\n";
    out << inputSize << " " << outputSize << "\n";
    
    file.close();
    return true;
}

bool LogisticModel::load(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return false;
    }
    
    QTextStream in(&file);
    QString modelType = in.readLine();
    if (modelType != "LogisticModel") {
        file.close();
        return false;
    }
    
    in >> inputSize >> outputSize;
    
    file.close();
    return true;
}

double LogisticModel::sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

QVector<double> LogisticModel::softmax(const QVector<double> &input)
{
    QVector<double> output(input.size());
    double maxVal = *std::max_element(input.begin(), input.end());
    double sum = 0.0;
    
    for (int i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - maxVal);
        sum += output[i];
    }
    
    for (int i = 0; i < output.size(); ++i) {
        output[i] /= sum;
    }
    
    return output;
}

// Random Forest implementation (simplified)
RandomForest::RandomForest()
    : inputSize(0), outputSize(0), numTrees(10)
{
}

RandomForest::~RandomForest()
{
    for (auto tree : trees) {
        cleanupTree(tree);
    }
    trees.clear();
}

void RandomForest::initialize(int inputSize, int outputSize)
{
    this->inputSize = inputSize;
    this->outputSize = outputSize;
    
    // Clear existing trees
    for (auto tree : trees) {
        cleanupTree(tree);
    }
    trees.clear();
}

QVector<double> RandomForest::predict(const QVector<double> &input)
{
    QVector<double> predictions(outputSize, 0.0);
    
    for (auto tree : trees) {
        double prediction = predictTree(tree, input);
        int classIndex = static_cast<int>(prediction + 0.5);
        if (classIndex >= 0 && classIndex < outputSize) {
            predictions[classIndex] += 1.0;
        }
    }
    
    // Normalize
    double sum = 0.0;
    for (double &pred : predictions) {
        sum += pred;
    }
    if (sum > 0) {
        for (double &pred : predictions) {
            pred /= sum;
        }
    }
    
    return predictions;
}

void RandomForest::train(const QVector<QVector<double>> &inputs, 
                        const QVector<QVector<double>> &targets, 
                        double learningRate)
{
    // Convert targets to labels
    QVector<int> labels(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
        // Find the class with highest probability
        int maxIndex = 0;
        double maxVal = targets[i][0];
        for (int j = 1; j < targets[i].size(); ++j) {
            if (targets[i][j] > maxVal) {
                maxVal = targets[i][j];
                maxIndex = j;
            }
        }
        labels[i] = maxIndex;
    }
    
    // Build trees
    for (int i = 0; i < numTrees; ++i) {
        DecisionTree *tree = buildTree(inputs, labels);
        trees.append(tree);
    }
}

bool RandomForest::save(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return false;
    }
    
    QTextStream out(&file);
    out << "RandomForest\n";
    out << inputSize << " " << outputSize << " " << numTrees << "\n";
    
    file.close();
    return true;
}

bool RandomForest::load(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return false;
    }
    
    QTextStream in(&file);
    QString modelType = in.readLine();
    if (modelType != "RandomForest") {
        file.close();
        return false;
    }
    
    in >> inputSize >> outputSize >> numTrees;
    
    file.close();
    return true;
}

RandomForest::DecisionTree* RandomForest::buildTree(const QVector<QVector<double>> &data, 
                                                   const QVector<int> &labels, 
                                                   int depth)
{
    if (data.isEmpty() || depth > 10) {
        return nullptr;
    }
    
    DecisionTree *tree = new DecisionTree();
    
    // Simple splitting criterion (in practice, you'd use more sophisticated methods)
    if (depth < 5 && data.size() > 10) {
        // Choose a random feature and threshold
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> featureDist(0, inputSize - 1);
        
        tree->featureIndex = featureDist(gen);
        
        // Find a threshold
        double minVal = data[0][tree->featureIndex];
        double maxVal = data[0][tree->featureIndex];
        for (const auto &sample : data) {
            minVal = std::min(minVal, sample[tree->featureIndex]);
            maxVal = std::max(maxVal, sample[tree->featureIndex]);
        }
        
        std::uniform_real_distribution<double> thresholdDist(minVal, maxVal);
        tree->threshold = thresholdDist(gen);
        
        // Split data
        QVector<QVector<double>> leftData, rightData;
        QVector<int> leftLabels, rightLabels;
        
        for (int i = 0; i < data.size(); ++i) {
            if (data[i][tree->featureIndex] < tree->threshold) {
                leftData.append(data[i]);
                leftLabels.append(labels[i]);
            } else {
                rightData.append(data[i]);
                rightLabels.append(labels[i]);
            }
        }
        
        // Recursively build children
        tree->left = buildTree(leftData, leftLabels, depth + 1);
        tree->right = buildTree(rightData, rightLabels, depth + 1);
    } else {
        // Leaf node - predict majority class
        QVector<int> classCounts(outputSize, 0);
        for (int label : labels) {
            if (label >= 0 && label < outputSize) {
                classCounts[label]++;
            }
        }
        
        int maxClass = 0;
        int maxCount = classCounts[0];
        for (int i = 1; i < outputSize; ++i) {
            if (classCounts[i] > maxCount) {
                maxCount = classCounts[i];
                maxClass = i;
            }
        }
        
        tree->value = static_cast<double>(maxClass);
    }
    
    return tree;
}

double RandomForest::predictTree(DecisionTree *tree, const QVector<double> &input)
{
    if (!tree) {
        return 0.0;
    }
    
    if (tree->featureIndex >= 0) {
        // Internal node
        if (input[tree->featureIndex] < tree->threshold) {
            return predictTree(tree->left, input);
        } else {
            return predictTree(tree->right, input);
        }
    } else {
        // Leaf node
        return tree->value;
    }
}

void RandomForest::cleanupTree(DecisionTree *tree)
{
    if (!tree) {
        return;
    }
    
    cleanupTree(tree->left);
    cleanupTree(tree->right);
    delete tree;
} 