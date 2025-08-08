#ifndef MODELAINTRAINER_H
#define MODELAINTRAINER_H

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <memory>
#include <functional>
#include <random>

// Forward declarations
class NeuralNetwork;
class LinearModel;
class LogisticModel;
class RandomForest;

struct TrainingConfig {
    std::string modelType;
    int inputSize;
    int hiddenSize;
    int outputSize;
    double learningRate;
    int batchSize;
    int epochs;
    std::string datasetPath;
    std::string outputPath;
    
    TrainingConfig() : inputSize(784), hiddenSize(128), outputSize(10), 
                      learningRate(0.001), batchSize(32), epochs(100) {}
};

struct TrainingMetrics {
    int currentEpoch;
    double loss;
    double accuracy;
    double validationLoss;
    double validationAccuracy;
    std::vector<double> lossHistory;
    std::vector<double> accuracyHistory;
    std::vector<double> validationLossHistory;
    std::vector<double> validationAccuracyHistory;
    
    TrainingMetrics() : currentEpoch(0), loss(0.0), accuracy(0.0), 
                       validationLoss(0.0), validationAccuracy(0.0) {}
};

class ModelTrainer
{
public:
    explicit ModelTrainer();
    ~ModelTrainer();

    // Configuration setters
    void setDatasetPath(const std::string& path);
    void setOutputPath(const std::string& path);
    void setModelType(const std::string& type);
    void setInputSize(int size);
    void setHiddenSize(int size);
    void setOutputSize(int size);
    void setLearningRate(double rate);
    void setBatchSize(int size);
    void setEpochs(int epochs);
    
    // Training control
    void startTraining();
    void stopTraining();
    void pauseTraining();
    void resumeTraining();
    
    // Model operations
    bool saveModel(const std::string& path);
    bool loadModel(const std::string& path);
    
    // Getters
    TrainingConfig getConfig() const;
    TrainingMetrics getMetrics() const;
    bool isTraining() const;
    bool isPaused() const;
    
    // Progress callback
    void setProgressCallback(std::function<void(int, double, double)> callback);

private:
    // Model creation and management
    void createModel();
    void initializeModel();
    void cleanupModel();
    
    // Training algorithms
    void trainNeuralNetwork();
    void trainLinearModel();
    void trainLogisticModel();
    void trainRandomForest();
    
    // Utility functions
    double calculateLoss(const std::vector<double>& predictions, const std::vector<double>& targets);
    double calculateAccuracy(const std::vector<double>& predictions, const std::vector<int>& targets);
    std::vector<double> forwardPass(const std::vector<double>& input);
    void backwardPass(const std::vector<double>& input, const std::vector<double>& target);
    
    // Data processing
    std::vector<std::vector<double>> getBatch(int batchIndex);
    std::vector<int> getBatchLabels(int batchIndex);
    void normalizeData();
    void shuffleData();
    
    // Model instances
    std::unique_ptr<NeuralNetwork> neuralNetwork;
    std::unique_ptr<LinearModel> linearModel;
    std::unique_ptr<LogisticModel> logisticModel;
    std::unique_ptr<RandomForest> randomForest;
    
    // Training state
    TrainingConfig config;
    TrainingMetrics metrics;
    std::vector<std::vector<double>> trainingData;
    std::vector<int> trainingLabels;
    std::vector<std::vector<double>> validationData;
    std::vector<int> validationLabels;
    
    // Control flags
    bool trainingFlag;
    bool pauseFlag;
    bool stopFlag;
    
    // Threading
    std::mutex trainingMutex;
    std::thread trainingThread;
    
    // Progress callback
    std::function<void(int, double, double)> progressCallback;
    
    // Random number generation
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist;
};

// Model base class
class BaseModel
{
public:
    virtual ~BaseModel() = default;
    virtual void initialize(int inputSize, int outputSize) = 0;
    virtual std::vector<double> predict(const std::vector<double>& input) = 0;
    virtual void train(const std::vector<std::vector<double>>& inputs, 
                      const std::vector<std::vector<double>>& targets, 
                      double learningRate) = 0;
    virtual bool save(const std::string& path) = 0;
    virtual bool load(const std::string& path) = 0;
};

// Neural Network implementation
class NeuralNetwork : public BaseModel
{
public:
    NeuralNetwork();
    ~NeuralNetwork();
    
    void initialize(int inputSize, int outputSize) override;
    std::vector<double> predict(const std::vector<double>& input) override;
    void train(const std::vector<std::vector<double>>& inputs, 
               const std::vector<std::vector<double>>& targets, 
               double learningRate) override;
    bool save(const std::string& path) override;
    bool load(const std::string& path) override;
    
    void setHiddenSize(int size);

private:
    std::vector<std::vector<double>> weights1; // Input to hidden
    std::vector<std::vector<double>> weights2; // Hidden to output
    std::vector<double> bias1;
    std::vector<double> bias2;
    int inputSize;
    int hiddenSize;
    int outputSize;
    
    double sigmoid(double x);
    double sigmoidDerivative(double x);
    std::vector<double> softmax(const std::vector<double>& input);
};

// Linear Regression model
class LinearModel : public BaseModel
{
public:
    LinearModel();
    ~LinearModel();
    
    void initialize(int inputSize, int outputSize) override;
    std::vector<double> predict(const std::vector<double>& input) override;
    void train(const std::vector<std::vector<double>>& inputs, 
               const std::vector<std::vector<double>>& targets, 
               double learningRate) override;
    bool save(const std::string& path) override;
    bool load(const std::string& path) override;

private:
    std::vector<std::vector<double>> weights;
    std::vector<double> bias;
    int inputSize;
    int outputSize;
};

// Logistic Regression model
class LogisticModel : public BaseModel
{
public:
    LogisticModel();
    ~LogisticModel();
    
    void initialize(int inputSize, int outputSize) override;
    std::vector<double> predict(const std::vector<double>& input) override;
    void train(const std::vector<std::vector<double>>& inputs, 
               const std::vector<std::vector<double>>& targets, 
               double learningRate) override;
    bool save(const std::string& path) override;
    bool load(const std::string& path) override;

private:
    std::vector<std::vector<double>> weights;
    std::vector<double> bias;
    int inputSize;
    int outputSize;
    
    double sigmoid(double x);
    std::vector<double> softmax(const std::vector<double>& input);
};

// Random Forest model (simplified)
class RandomForest : public BaseModel
{
public:
    RandomForest();
    ~RandomForest();
    
    void initialize(int inputSize, int outputSize) override;
    std::vector<double> predict(const std::vector<double>& input) override;
    void train(const std::vector<std::vector<double>>& inputs, 
               const std::vector<std::vector<double>>& targets, 
               double learningRate) override;
    bool save(const std::string& path) override;
    bool load(const std::string& path) override;

private:
    struct DecisionTree {
        int featureIndex;
        double threshold;
        double value;
        DecisionTree* left;
        DecisionTree* right;
        
        DecisionTree() : featureIndex(-1), threshold(0.0), value(0.0), left(nullptr), right(nullptr) {}
    };
    
    std::vector<DecisionTree*> trees;
    int inputSize;
    int outputSize;
    int numTrees;
    
    DecisionTree* buildTree(const std::vector<std::vector<double>>& data, 
                           const std::vector<int>& labels, 
                           int depth = 0);
    double predictTree(DecisionTree* tree, const std::vector<double>& input);
    void cleanupTree(DecisionTree* tree);
};

#endif // MODELAINTRAINER_H 