#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QTextEdit>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QProgressBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QTableWidget>
#include <QHeaderView>
#include <QMenuBar>
#include <QStatusBar>
#include <QTimer>
#include <QThread>
#include <QMutex>
#include <QFuture>
#include <QtConcurrent>
#include <QChartView>
#include <QChart>
#include <QSplineSeries>
#include <QValueAxis>
#include <QCategoryAxis>

#include "datasetmanager.h"
#include "modeltrainer.h"
#include "progressdialog.h"
#include "chartwidget.h"

QT_CHARTS_USE_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void browseDataset();
    void browseOutputPath();
    void startTraining();
    void stopTraining();
    void saveModel();
    void loadModel();
    void updateProgress(int epoch, double loss, double accuracy);
    void trainingFinished();
    void showAbout();
    void showHelp();

private:
    void setupUI();
    void setupMenuBar();
    void setupStatusBar();
    void connectSignals();
    void updateDatasetInfo();
    void updateTrainingStatus(bool isTraining);
    void logMessage(const QString &message);

    // UI Components
    QWidget *centralWidget;
    QVBoxLayout *mainLayout;
    
    // Dataset Section
    QGroupBox *datasetGroup;
    QGridLayout *datasetLayout;
    QLabel *datasetPathLabel;
    QLineEdit *datasetPathEdit;
    QPushButton *browseDatasetBtn;
    QLabel *datasetInfoLabel;
    QTableWidget *datasetPreviewTable;
    
    // Model Configuration Section
    QGroupBox *modelConfigGroup;
    QGridLayout *modelConfigLayout;
    QLabel *modelTypeLabel;
    QComboBox *modelTypeCombo;
    QLabel *inputSizeLabel;
    QSpinBox *inputSizeSpin;
    QLabel *hiddenSizeLabel;
    QSpinBox *hiddenSizeSpin;
    QLabel *outputSizeLabel;
    QSpinBox *outputSizeSpin;
    QLabel *learningRateLabel;
    QDoubleSpinBox *learningRateSpin;
    QLabel *batchSizeLabel;
    QSpinBox *batchSizeSpin;
    QLabel *epochsLabel;
    QSpinBox *epochsSpin;
    
    // Training Section
    QGroupBox *trainingGroup;
    QVBoxLayout *trainingLayout;
    QHBoxLayout *trainingButtonsLayout;
    QPushButton *startTrainingBtn;
    QPushButton *stopTrainingBtn;
    QPushButton *saveModelBtn;
    QPushButton *loadModelBtn;
    QProgressBar *trainingProgressBar;
    QLabel *currentEpochLabel;
    QLabel *currentLossLabel;
    QLabel *currentAccuracyLabel;
    
    // Output Section
    QGroupBox *outputGroup;
    QGridLayout *outputLayout;
    QLabel *outputPathLabel;
    QLineEdit *outputPathEdit;
    QPushButton *browseOutputBtn;
    QTextEdit *logTextEdit;
    
    // Charts Section
    QGroupBox *chartsGroup;
    QVBoxLayout *chartsLayout;
    ChartWidget *lossChart;
    ChartWidget *accuracyChart;
    
    // Menu and Status
    QMenuBar *menuBar;
    QStatusBar *statusBar;
    
    // Data members
    DatasetManager *datasetManager;
    ModelTrainer *modelTrainer;
    QThread *trainingThread;
    QTimer *updateTimer;
    bool isTraining;
    
    // Training data for charts
    QVector<double> lossHistory;
    QVector<double> accuracyHistory;
    QVector<int> epochHistory;
};

#endif // MAINWINDOW_H 