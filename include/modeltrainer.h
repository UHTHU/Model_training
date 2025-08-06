#ifndef MODELAINTRAINER_H
#define MODELAINTRAINER_H

#include <QObject>
#include <QString>
#include <QVector>
#include <QThread>
#include <QMutex>
#include <QTimer>
#include <random>
#include <memory>

// Forward declarations
class NeuralNetwork;
class LinearModel;
class LogisticModel;
class RandomForest;

struct TrainingConfig {
    QString modelType;
    int inputSize;
    int hiddenSize;
    int outputSize;
    double learningRate;
    int batchSize;
    int epochs;
    QString datasetPath;
    QString outputPath;
    
    TrainingConfig() : inputSize(784), hiddenSize(128), outputSize(10), 
                      learningRate(0.001), batchSize(32), epochs(100) {}
};

struct TrainingMetrics {
    int currentEpoch;
    double loss;
    double accuracy;
    double validationLoss;
    double validationAccuracy;
    QVector<double> lossHistory;
    QVector<double> accuracyHistory;
    QVector<double> validationLossHistory;
    QVector<double> validationAccuracyHistory;
    
    TrainingMetrics() : currentEpoch(0), loss(0.0), accuracy(0.0), 
                       validationLoss(0.0), validationAccuracy(0.0) {}
};

class ModelTrainer : public QObject
{
    Q_OBJECT

public:
    explicit ModelTrainer(QObject *parent = nullptr);
    ~ModelTrainer();

    // Configuration setters
    void setDatasetPath(const QString &path);
    void setOutputPath(const QString &path);
    void setModelType(const QString &type);
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
    bool saveModel(const QString &path);
    bool loadModel(const QString &path);
    
    // Getters
    TrainingConfig getConfig() const;
    TrainingMetrics getMetrics() const;
    bool isTraining() const;
    bool isPaused() const;

signals:
    void progressUpdated(int epoch, double loss, double accuracy);
    void trainingFinished();
    void trainingError(const QString &error);
    void modelSaved(const QString &path);
    void modelLoaded(const QString &path);

private slots:
    void trainingLoop();

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
    double calculateLoss(const QVector<double> &predictions, const QVector<double> &targets);
    double calculateAccuracy(const QVector<double> &predictions, const QVector<int> &targets);
    QVector<double> forwardPass(const QVector<double> &input);
    void backwardPass(const QVector<double> &input, const QVector<double> &target);
    
    // Data processing
    QVector<QVector<double>> getBatch(int batchIndex);
    QVector<int> getBatchLabels(int batchIndex);
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
    QVector<QVector<double>> trainingData;
    QVector<int> trainingLabels;
    QVector<QVector<double>> validationData;
    QVector<int> validationLabels;
    
    // Control flags
    bool trainingFlag;
    bool pauseFlag;
    bool stopFlag;
    
    // Threading
    QMutex trainingMutex;
    QTimer *trainingTimer;
    
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
    virtual QVector<double> predict(const QVector<double> &input) = 0;
    virtual void train(const QVector<QVector<double>> &inputs, 
                      const QVector<QVector<double>> &targets, 
                      double learningRate) = 0;
    virtual bool save(const QString &path) = 0;
    virtual bool load(const QString &path) = 0;
};

// Neural Network implementation
class NeuralNetwork : public BaseModel
{
public:
    NeuralNetwork();
    ~NeuralNetwork();
    
    void initialize(int inputSize, int outputSize) override;
    QVector<double> predict(const QVector<double> &input) override;
    void train(const QVector<QVector<double>> &inputs, 
               const QVector<QVector<double>> &targets, 
               double learningRate) override;
    bool save(const QString &path) override;
    bool load(const QString &path) override;
    
    void setHiddenSize(int size);

private:
    QVector<QVector<double>> weights1; // Input to hidden
    QVector<QVector<double>> weights2; // Hidden to output
    QVector<double> bias1;
    QVector<double> bias2;
    int inputSize;
    int hiddenSize;
    int outputSize;
    
    double sigmoid(double x);
    double sigmoidDerivative(double x);
    QVector<double> softmax(const QVector<double> &input);
};

// Linear Regression model
class LinearModel : public BaseModel
{
public:
    LinearModel();
    ~LinearModel();
    
    void initialize(int inputSize, int outputSize) override;
    QVector<double> predict(const QVector<double> &input) override;
    void train(const QVector<QVector<double>> &inputs, 
               const QVector<QVector<double>> &targets, 
               double learningRate) override;
    bool save(const QString &path) override;
    bool load(const QString &path) override;

private:
    QVector<QVector<double>> weights;
    QVector<double> bias;
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
    QVector<double> predict(const QVector<double> &input) override;
    void train(const QVector<QVector<double>> &inputs, 
               const QVector<QVector<double>> &targets, 
               double learningRate) override;
    bool save(const QString &path) override;
    bool load(const QString &path) override;

private:
    QVector<QVector<double>> weights;
    QVector<double> bias;
    int inputSize;
    int outputSize;
    
    double sigmoid(double x);
    QVector<double> softmax(const QVector<double> &input);
};

// Random Forest model (simplified)
class RandomForest : public BaseModel
{
public:
    RandomForest();
    ~RandomForest();
    
    void initialize(int inputSize, int outputSize) override;
    QVector<double> predict(const QVector<double> &input) override;
    void train(const QVector<QVector<double>> &inputs, 
               const QVector<QVector<double>> &targets, 
               double learningRate) override;
    bool save(const QString &path) override;
    bool load(const QString &path) override;

private:
    struct DecisionTree {
        int featureIndex;
        double threshold;
        double value;
        DecisionTree *left;
        DecisionTree *right;
        
        DecisionTree() : featureIndex(-1), threshold(0.0), value(0.0), left(nullptr), right(nullptr) {}
    };
    
    QVector<DecisionTree*> trees;
    int inputSize;
    int outputSize;
    int numTrees;
    
    DecisionTree* buildTree(const QVector<QVector<double>> &data, 
                           const QVector<int> &labels, 
                           int depth = 0);
    double predictTree(DecisionTree *tree, const QVector<double> &input);
    void cleanupTree(DecisionTree *tree);
};

#endif // MODELAINTRAINER_H 