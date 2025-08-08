#ifndef DATASETMANAGER_H
#define DATASETMANAGER_H

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <memory>
#include <mutex>

struct DatasetInfo {
    std::string path;
    std::string name;
    int totalSamples;
    int inputFeatures;
    int outputClasses;
    std::string format;
    bool isValid;
    std::string errorMessage;
    
    DatasetInfo() : totalSamples(0), inputFeatures(0), outputClasses(0), isValid(false) {}
};

struct DataSample {
    std::vector<double> features;
    int label;
    std::string type; // "training", "validation", "test"
    
    DataSample() : label(0) {}
};

class DatasetManager
{
public:
    explicit DatasetManager();
    ~DatasetManager();

    bool loadDataset(const std::string& path);
    bool validateDataset(const std::string& path);
    DatasetInfo getDatasetInfo() const;
    std::vector<DataSample> getTrainingSamples() const;
    std::vector<DataSample> getValidationSamples() const;
    std::vector<DataSample> getTestSamples() const;
    std::vector<DataSample> getSamples(int start, int count) const;
    int getTotalSamples() const;
    int getInputFeatures() const;
    int getOutputClasses() const;
    bool isDatasetLoaded() const;
    void clearDataset();

private:
    bool loadCSVDataset(const std::string& path);
    bool loadImageDataset(const std::string& path);
    bool loadTextDataset(const std::string& path);
    bool loadCustomDataset(const std::string& path);
    
    bool parseCSVLine(const std::string& line, DataSample& sample);
    bool detectDatasetFormat(const std::string& path);
    void updateDatasetInfo();
    
    DatasetInfo datasetInfo;
    std::vector<DataSample> trainingSamples;
    std::vector<DataSample> validationSamples;
    std::vector<DataSample> testSamples;
    
    mutable std::mutex dataMutex;
    bool datasetLoadedFlag;
    
    // Supported formats
    std::vector<std::string> supportedFormats;
};

#endif // DATASETMANAGER_H 