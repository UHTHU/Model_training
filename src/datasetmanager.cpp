#include "datasetmanager.h"
#include <QFile>
#include <QTextStream>
#include <QDirIterator>
#include <QDebug>
#include <QRegularExpression>
#include <QSet>

DatasetManager::DatasetManager(QObject *parent)
    : QObject(parent)
    , datasetLoadedFlag(false)
{
    // Initialize supported formats
    supportedFormats << "csv" << "txt" << "json" << "xml" << "h5" << "hdf5" << "mat" << "npy" << "npz";
    
    // Initialize dataset info
    datasetInfo.isValid = false;
}

DatasetManager::~DatasetManager()
{
    clearDataset();
}

bool DatasetManager::loadDataset(const QString &path)
{
    QMutexLocker locker(&dataMutex);
    
    // Clear previous dataset
    clearDataset();
    
    // Validate path
    QFileInfo fileInfo(path);
    if (!fileInfo.exists()) {
        emit datasetLoadError("Dataset path does not exist: " + path);
        return false;
    }
    
    // Detect format and load accordingly
    QString format = detectDatasetFormat(path);
    bool success = false;
    
    if (format == "csv" || format == "txt") {
        success = loadCSVDataset(path);
    } else if (format == "image") {
        success = loadImageDataset(path);
    } else if (format == "text") {
        success = loadTextDataset(path);
    } else {
        success = loadCustomDataset(path);
    }
    
    if (success) {
        datasetInfo.path = path;
        datasetInfo.name = fileInfo.baseName();
        datasetInfo.format = format;
        datasetInfo.isValid = true;
        datasetLoadedFlag = true;
        
        updateDatasetInfo();
        emit datasetLoaded(datasetInfo);
        
        qDebug() << "Dataset loaded successfully:" << path;
        qDebug() << "Total samples:" << datasetInfo.totalSamples;
        qDebug() << "Input features:" << datasetInfo.inputFeatures;
        qDebug() << "Output classes:" << datasetInfo.outputClasses;
    } else {
        datasetInfo.errorMessage = "Failed to load dataset";
        emit datasetLoadError(datasetInfo.errorMessage);
    }
    
    return success;
}

bool DatasetManager::validateDataset(const QString &path)
{
    QFileInfo fileInfo(path);
    if (!fileInfo.exists()) {
        emit datasetValidated(false, "Dataset path does not exist");
        return false;
    }
    
    // Basic validation - check if we can read the file/directory
    bool isValid = false;
    QString message;
    
    if (fileInfo.isFile()) {
        QFile file(path);
        if (file.open(QIODevice::ReadOnly)) {
            isValid = true;
            message = "File is readable";
            file.close();
        } else {
            message = "Cannot read file: " + file.errorString();
        }
    } else if (fileInfo.isDir()) {
        QDir dir(path);
        if (dir.exists() && dir.isReadable()) {
            isValid = true;
            message = "Directory is readable";
        } else {
            message = "Cannot read directory";
        }
    } else {
        message = "Path is neither a file nor directory";
    }
    
    emit datasetValidated(isValid, message);
    return isValid;
}

DatasetInfo DatasetManager::getDatasetInfo() const
{
    QMutexLocker locker(&dataMutex);
    return datasetInfo;
}

QVector<DataSample> DatasetManager::getTrainingSamples() const
{
    QMutexLocker locker(&dataMutex);
    return trainingSamples;
}

QVector<DataSample> DatasetManager::getValidationSamples() const
{
    QMutexLocker locker(&dataMutex);
    return validationSamples;
}

QVector<DataSample> DatasetManager::getTestSamples() const
{
    QMutexLocker locker(&dataMutex);
    return testSamples;
}

QVector<DataSample> DatasetManager::getSamples(int start, int count) const
{
    QMutexLocker locker(&dataMutex);
    QVector<DataSample> samples;
    
    int totalSamples = trainingSamples.size() + validationSamples.size() + testSamples.size();
    if (start >= totalSamples || count <= 0) {
        return samples;
    }
    
    // Combine all samples
    QVector<DataSample> allSamples;
    allSamples.append(trainingSamples);
    allSamples.append(validationSamples);
    allSamples.append(testSamples);
    
    int end = qMin(start + count, totalSamples);
    for (int i = start; i < end; ++i) {
        samples.append(allSamples[i]);
    }
    
    return samples;
}

int DatasetManager::getTotalSamples() const
{
    QMutexLocker locker(&dataMutex);
    return datasetInfo.totalSamples;
}

int DatasetManager::getInputFeatures() const
{
    QMutexLocker locker(&dataMutex);
    return datasetInfo.inputFeatures;
}

int DatasetManager::getOutputClasses() const
{
    QMutexLocker locker(&dataMutex);
    return datasetInfo.outputClasses;
}

bool DatasetManager::isDatasetLoaded() const
{
    QMutexLocker locker(&dataMutex);
    return datasetLoadedFlag;
}

void DatasetManager::clearDataset()
{
    QMutexLocker locker(&dataMutex);
    
    trainingSamples.clear();
    validationSamples.clear();
    testSamples.clear();
    
    datasetInfo = DatasetInfo();
    datasetLoadedFlag = false;
}

bool DatasetManager::loadCSVDataset(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        datasetInfo.errorMessage = "Cannot open CSV file: " + file.errorString();
        return false;
    }
    
    QTextStream in(&file);
    QString line;
    int lineCount = 0;
    QSet<int> uniqueLabels;
    
    // Read header (optional)
    if (!in.atEnd()) {
        line = in.readLine();
        lineCount++;
        // Skip header if it doesn't contain numbers
        if (!line.contains(QRegularExpression("\\d"))) {
            // This is a header, skip it
        } else {
            // This is data, process it
            DataSample sample;
            parseCSVLine(line, sample);
            if (sample.features.size() > 0) {
                trainingSamples.append(sample);
                uniqueLabels.insert(sample.label);
            }
        }
    }
    
    // Read data lines
    while (!in.atEnd()) {
        line = in.readLine();
        lineCount++;
        
        if (line.trimmed().isEmpty()) {
            continue;
        }
        
        DataSample sample;
        parseCSVLine(line, sample);
        
        if (sample.features.size() > 0) {
            // Determine sample type based on line number or other criteria
            if (lineCount % 10 == 0) {
                sample.type = "validation";
                validationSamples.append(sample);
            } else if (lineCount % 20 == 0) {
                sample.type = "test";
                testSamples.append(sample);
            } else {
                sample.type = "training";
                trainingSamples.append(sample);
            }
            
            uniqueLabels.insert(sample.label);
        }
    }
    
    file.close();
    
    // Update dataset info
    datasetInfo.totalSamples = trainingSamples.size() + validationSamples.size() + testSamples.size();
    if (datasetInfo.totalSamples > 0) {
        datasetInfo.inputFeatures = trainingSamples[0].features.size();
        datasetInfo.outputClasses = uniqueLabels.size();
    }
    
    return datasetInfo.totalSamples > 0;
}

bool DatasetManager::loadImageDataset(const QString &path)
{
    QDir dir(path);
    if (!dir.exists()) {
        datasetInfo.errorMessage = "Image directory does not exist";
        return false;
    }
    
    // Look for common image subdirectory structures
    QStringList imageExtensions = {"*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"};
    QSet<int> uniqueLabels;
    
    // Check for class-based directory structure
    QDirIterator dirIter(path, QDir::Dirs | QDir::NoDotAndDotDot, QDirIterator::Subdirectories);
    bool hasClassDirs = false;
    
    while (dirIter.hasNext()) {
        QString subDir = dirIter.next();
        QDir classDir(subDir);
        QString className = classDir.dirName();
        
        // Check if this looks like a class directory (contains images)
        QStringList images = classDir.entryList(imageExtensions, QDir::Files);
        if (!images.isEmpty()) {
            hasClassDirs = true;
            bool ok;
            int label = className.toInt(&ok);
            if (!ok) {
                // Use hash of class name as label
                label = qHash(className) % 1000;
            }
            uniqueLabels.insert(label);
            
            // Add samples for this class
            for (const QString &imageFile : images) {
                DataSample sample;
                sample.label = label;
                sample.type = "training";
                
                // For now, create dummy features (in real implementation, you'd load and process the image)
                sample.features.resize(784); // 28x28 image flattened
                for (int i = 0; i < 784; ++i) {
                    sample.features[i] = (qrand() % 256) / 255.0; // Random pixel values
                }
                
                trainingSamples.append(sample);
            }
        }
    }
    
    if (!hasClassDirs) {
        // Try to load images directly from the directory
        QStringList images = dir.entryList(imageExtensions, QDir::Files);
        for (const QString &imageFile : images) {
            DataSample sample;
            sample.label = 0; // Default label
            sample.type = "training";
            
            // Create dummy features
            sample.features.resize(784);
            for (int i = 0; i < 784; ++i) {
                sample.features[i] = (qrand() % 256) / 255.0;
            }
            
            trainingSamples.append(sample);
        }
    }
    
    // Update dataset info
    datasetInfo.totalSamples = trainingSamples.size();
    if (datasetInfo.totalSamples > 0) {
        datasetInfo.inputFeatures = 784; // Standard image size
        datasetInfo.outputClasses = uniqueLabels.size();
        if (datasetInfo.outputClasses == 0) {
            datasetInfo.outputClasses = 1; // At least one class
        }
    }
    
    return datasetInfo.totalSamples > 0;
}

bool DatasetManager::loadTextDataset(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        datasetInfo.errorMessage = "Cannot open text file: " + file.errorString();
        return false;
    }
    
    QTextStream in(&file);
    QString line;
    int lineCount = 0;
    QSet<int> uniqueLabels;
    
    while (!in.atEnd()) {
        line = in.readLine();
        lineCount++;
        
        if (line.trimmed().isEmpty()) {
            continue;
        }
        
        // Simple text processing - split by whitespace
        QStringList parts = line.split(QRegularExpression("\\s+"), QString::SkipEmptyParts);
        if (parts.size() < 2) {
            continue;
        }
        
        DataSample sample;
        
        // Last part is the label
        bool ok;
        sample.label = parts.last().toInt(&ok);
        if (!ok) {
            sample.label = 0;
        }
        uniqueLabels.insert(sample.label);
        
        // Rest are features
        sample.features.resize(parts.size() - 1);
        for (int i = 0; i < parts.size() - 1; ++i) {
            sample.features[i] = parts[i].toDouble(&ok);
            if (!ok) {
                sample.features[i] = 0.0;
            }
        }
        
        // Determine sample type
        if (lineCount % 10 == 0) {
            sample.type = "validation";
            validationSamples.append(sample);
        } else if (lineCount % 20 == 0) {
            sample.type = "test";
            testSamples.append(sample);
        } else {
            sample.type = "training";
            trainingSamples.append(sample);
        }
    }
    
    file.close();
    
    // Update dataset info
    datasetInfo.totalSamples = trainingSamples.size() + validationSamples.size() + testSamples.size();
    if (datasetInfo.totalSamples > 0) {
        datasetInfo.inputFeatures = trainingSamples[0].features.size();
        datasetInfo.outputClasses = uniqueLabels.size();
    }
    
    return datasetInfo.totalSamples > 0;
}

bool DatasetManager::loadCustomDataset(const QString &path)
{
    // Placeholder for custom dataset formats
    // This could be extended to support HDF5, NumPy arrays, etc.
    datasetInfo.errorMessage = "Custom dataset format not yet implemented";
    return false;
}

void DatasetManager::parseCSVLine(const QString &line, DataSample &sample)
{
    QStringList parts = line.split(',');
    if (parts.size() < 2) {
        return;
    }
    
    // Last part is the label
    bool ok;
    sample.label = parts.last().toInt(&ok);
    if (!ok) {
        sample.label = 0;
    }
    
    // Rest are features
    sample.features.resize(parts.size() - 1);
    for (int i = 0; i < parts.size() - 1; ++i) {
        sample.features[i] = parts[i].toDouble(&ok);
        if (!ok) {
            sample.features[i] = 0.0;
        }
    }
}

bool DatasetManager::detectDatasetFormat(const QString &path)
{
    QFileInfo fileInfo(path);
    QString extension = fileInfo.suffix().toLower();
    
    if (supportedFormats.contains(extension)) {
        return true;
    }
    
    // Check if it's a directory (might contain images)
    if (fileInfo.isDir()) {
        QDir dir(path);
        QStringList imageExtensions = {"*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"};
        QStringList images = dir.entryList(imageExtensions, QDir::Files);
        if (!images.isEmpty()) {
            return true;
        }
    }
    
    return false;
}

void DatasetManager::updateDatasetInfo()
{
    datasetInfo.totalSamples = trainingSamples.size() + validationSamples.size() + testSamples.size();
    
    if (trainingSamples.size() > 0) {
        datasetInfo.inputFeatures = trainingSamples[0].features.size();
    }
    
    // Count unique labels
    QSet<int> uniqueLabels;
    for (const DataSample &sample : trainingSamples) {
        uniqueLabels.insert(sample.label);
    }
    for (const DataSample &sample : validationSamples) {
        uniqueLabels.insert(sample.label);
    }
    for (const DataSample &sample : testSamples) {
        uniqueLabels.insert(sample.label);
    }
    
    datasetInfo.outputClasses = uniqueLabels.size();
} 