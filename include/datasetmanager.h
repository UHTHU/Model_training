#ifndef DATASETMANAGER_H
#define DATASETMANAGER_H

#include <QObject>
#include <QString>
#include <QVector>
#include <QDir>
#include <QFileInfo>
#include <QMutex>

struct DatasetInfo {
    QString path;
    QString name;
    int totalSamples;
    int inputFeatures;
    int outputClasses;
    QString format;
    bool isValid;
    QString errorMessage;
    
    DatasetInfo() : totalSamples(0), inputFeatures(0), outputClasses(0), isValid(false) {}
};

struct DataSample {
    QVector<double> features;
    int label;
    QString type; // "training", "validation", "test"
    
    DataSample() : label(0) {}
};

class DatasetManager : public QObject
{
    Q_OBJECT

public:
    explicit DatasetManager(QObject *parent = nullptr);
    ~DatasetManager();

    bool loadDataset(const QString &path);
    bool validateDataset(const QString &path);
    DatasetInfo getDatasetInfo() const;
    QVector<DataSample> getTrainingSamples() const;
    QVector<DataSample> getValidationSamples() const;
    QVector<DataSample> getTestSamples() const;
    QVector<DataSample> getSamples(int start, int count) const;
    int getTotalSamples() const;
    int getInputFeatures() const;
    int getOutputClasses() const;
    bool isDatasetLoaded() const;
    void clearDataset();

signals:
    void datasetLoaded(const DatasetInfo &info);
    void datasetLoadError(const QString &error);
    void datasetValidated(bool isValid, const QString &message);

private:
    bool loadCSVDataset(const QString &path);
    bool loadImageDataset(const QString &path);
    bool loadTextDataset(const QString &path);
    bool loadCustomDataset(const QString &path);
    
    void parseCSVLine(const QString &line, DataSample &sample);
    bool detectDatasetFormat(const QString &path);
    void updateDatasetInfo();
    
    DatasetInfo datasetInfo;
    QVector<DataSample> trainingSamples;
    QVector<DataSample> validationSamples;
    QVector<DataSample> testSamples;
    
    mutable QMutex dataMutex;
    bool datasetLoadedFlag;
    
    // Supported formats
    QStringList supportedFormats;
};

#endif // DATASETMANAGER_H 