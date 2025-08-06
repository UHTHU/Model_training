#include "mainwindow.h"
#include <QApplication>
#include <QDesktopWidget>
#include <QScreen>
#include <QDateTime>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , isTraining(false)
{
    // Initialize components
    datasetManager = new DatasetManager(this);
    modelTrainer = new ModelTrainer(this);
    trainingThread = new QThread(this);
    updateTimer = new QTimer(this);
    
    // Move trainer to separate thread
    modelTrainer->moveToThread(trainingThread);
    
    setupUI();
    setupMenuBar();
    setupStatusBar();
    connectSignals();
    
    // Set window properties
    setWindowTitle("Model Training Application");
    setMinimumSize(1200, 800);
    
    // Center window on screen
    QScreen *screen = QApplication::primaryScreen();
    QRect screenGeometry = screen->geometry();
    int x = (screenGeometry.width() - width()) / 2;
    int y = (screenGeometry.height() - height()) / 2;
    move(x, y);
    
    // Initialize training data
    lossHistory.clear();
    accuracyHistory.clear();
    epochHistory.clear();
    
    logMessage("Application started. Ready to load dataset and configure model.");
}

MainWindow::~MainWindow()
{
    if (isTraining) {
        stopTraining();
    }
    trainingThread->quit();
    trainingThread->wait();
}

void MainWindow::setupUI()
{
    centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    
    mainLayout = new QVBoxLayout(centralWidget);
    
    // Dataset Section
    datasetGroup = new QGroupBox("Dataset Configuration", centralWidget);
    datasetLayout = new QGridLayout(datasetGroup);
    
    datasetPathLabel = new QLabel("Dataset Path:", datasetGroup);
    datasetPathEdit = new QLineEdit(datasetGroup);
    datasetPathEdit->setPlaceholderText("Select dataset file or directory...");
    browseDatasetBtn = new QPushButton("Browse", datasetGroup);
    
    datasetInfoLabel = new QLabel("No dataset loaded", datasetGroup);
    datasetInfoLabel->setStyleSheet("color: gray; font-style: italic;");
    
    datasetPreviewTable = new QTableWidget(datasetGroup);
    datasetPreviewTable->setMaximumHeight(150);
    datasetPreviewTable->setColumnCount(5);
    datasetPreviewTable->setHorizontalHeaderLabels({"Sample", "Features", "Label", "Type", "Status"});
    datasetPreviewTable->horizontalHeader()->setStretchLastSection(true);
    
    datasetLayout->addWidget(datasetPathLabel, 0, 0);
    datasetLayout->addWidget(datasetPathEdit, 0, 1);
    datasetLayout->addWidget(browseDatasetBtn, 0, 2);
    datasetLayout->addWidget(datasetInfoLabel, 1, 0, 1, 3);
    datasetLayout->addWidget(datasetPreviewTable, 2, 0, 1, 3);
    
    // Model Configuration Section
    modelConfigGroup = new QGroupBox("Model Configuration", centralWidget);
    modelConfigLayout = new QGridLayout(modelConfigGroup);
    
    modelTypeLabel = new QLabel("Model Type:", modelConfigGroup);
    modelTypeCombo = new QComboBox(modelConfigGroup);
    modelTypeCombo->addItems({"Neural Network", "Linear Regression", "Logistic Regression", "Random Forest"});
    
    inputSizeLabel = new QLabel("Input Size:", modelConfigGroup);
    inputSizeSpin = new QSpinBox(modelConfigGroup);
    inputSizeSpin->setRange(1, 10000);
    inputSizeSpin->setValue(784);
    
    hiddenSizeLabel = new QLabel("Hidden Size:", modelConfigGroup);
    hiddenSizeSpin = new QSpinBox(modelConfigGroup);
    hiddenSizeSpin->setRange(1, 10000);
    hiddenSizeSpin->setValue(128);
    
    outputSizeLabel = new QLabel("Output Size:", modelConfigGroup);
    outputSizeSpin = new QSpinBox(modelConfigGroup);
    outputSizeSpin->setRange(1, 1000);
    outputSizeSpin->setValue(10);
    
    learningRateLabel = new QLabel("Learning Rate:", modelConfigGroup);
    learningRateSpin = new QDoubleSpinBox(modelConfigGroup);
    learningRateSpin->setRange(0.0001, 1.0);
    learningRateSpin->setValue(0.001);
    learningRateSpin->setDecimals(4);
    learningRateSpin->setSingleStep(0.0001);
    
    batchSizeLabel = new QLabel("Batch Size:", modelConfigGroup);
    batchSizeSpin = new QSpinBox(modelConfigGroup);
    batchSizeSpin->setRange(1, 10000);
    batchSizeSpin->setValue(32);
    
    epochsLabel = new QLabel("Epochs:", modelConfigGroup);
    epochsSpin = new QSpinBox(modelConfigGroup);
    epochsSpin->setRange(1, 10000);
    epochsSpin->setValue(100);
    
    modelConfigLayout->addWidget(modelTypeLabel, 0, 0);
    modelConfigLayout->addWidget(modelTypeCombo, 0, 1);
    modelConfigLayout->addWidget(inputSizeLabel, 0, 2);
    modelConfigLayout->addWidget(inputSizeSpin, 0, 3);
    modelConfigLayout->addWidget(hiddenSizeLabel, 1, 0);
    modelConfigLayout->addWidget(hiddenSizeSpin, 1, 1);
    modelConfigLayout->addWidget(outputSizeLabel, 1, 2);
    modelConfigLayout->addWidget(outputSizeSpin, 1, 3);
    modelConfigLayout->addWidget(learningRateLabel, 2, 0);
    modelConfigLayout->addWidget(learningRateSpin, 2, 1);
    modelConfigLayout->addWidget(batchSizeLabel, 2, 2);
    modelConfigLayout->addWidget(batchSizeSpin, 2, 3);
    modelConfigLayout->addWidget(epochsLabel, 3, 0);
    modelConfigLayout->addWidget(epochsSpin, 3, 1);
    
    // Training Section
    trainingGroup = new QGroupBox("Training Control", centralWidget);
    trainingLayout = new QVBoxLayout(trainingGroup);
    
    trainingButtonsLayout = new QHBoxLayout();
    startTrainingBtn = new QPushButton("Start Training", trainingGroup);
    startTrainingBtn->setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }");
    stopTrainingBtn = new QPushButton("Stop Training", trainingGroup);
    stopTrainingBtn->setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }");
    stopTrainingBtn->setEnabled(false);
    saveModelBtn = new QPushButton("Save Model", trainingGroup);
    loadModelBtn = new QPushButton("Load Model", trainingGroup);
    
    trainingProgressBar = new QProgressBar(trainingGroup);
    trainingProgressBar->setRange(0, 100);
    trainingProgressBar->setValue(0);
    
    currentEpochLabel = new QLabel("Epoch: 0/0", trainingGroup);
    currentLossLabel = new QLabel("Loss: 0.0000", trainingGroup);
    currentAccuracyLabel = new QLabel("Accuracy: 0.00%", trainingGroup);
    
    trainingButtonsLayout->addWidget(startTrainingBtn);
    trainingButtonsLayout->addWidget(stopTrainingBtn);
    trainingButtonsLayout->addWidget(saveModelBtn);
    trainingButtonsLayout->addWidget(loadModelBtn);
    trainingButtonsLayout->addStretch();
    
    trainingLayout->addLayout(trainingButtonsLayout);
    trainingLayout->addWidget(trainingProgressBar);
    trainingLayout->addWidget(currentEpochLabel);
    trainingLayout->addWidget(currentLossLabel);
    trainingLayout->addWidget(currentAccuracyLabel);
    
    // Output Section
    outputGroup = new QGroupBox("Output & Logging", centralWidget);
    outputLayout = new QGridLayout(outputGroup);
    
    outputPathLabel = new QLabel("Output Path:", outputGroup);
    outputPathEdit = new QLineEdit(outputGroup);
    outputPathEdit->setPlaceholderText("Select output directory...");
    browseOutputBtn = new QPushButton("Browse", outputGroup);
    
    logTextEdit = new QTextEdit(outputGroup);
    logTextEdit->setMaximumHeight(150);
    logTextEdit->setReadOnly(true);
    logTextEdit->setFont(QFont("Consolas", 9));
    
    outputLayout->addWidget(outputPathLabel, 0, 0);
    outputLayout->addWidget(outputPathEdit, 0, 1);
    outputLayout->addWidget(browseOutputBtn, 0, 2);
    outputLayout->addWidget(logTextEdit, 1, 0, 1, 3);
    
    // Charts Section
    chartsGroup = new QGroupBox("Training Progress", centralWidget);
    chartsLayout = new QVBoxLayout(chartsGroup);
    
    lossChart = new ChartWidget("Training Loss", chartsGroup);
    accuracyChart = new ChartWidget("Training Accuracy", chartsGroup);
    
    chartsLayout->addWidget(lossChart);
    chartsLayout->addWidget(accuracyChart);
    
    // Add all sections to main layout
    mainLayout->addWidget(datasetGroup);
    mainLayout->addWidget(modelConfigGroup);
    mainLayout->addWidget(trainingGroup);
    mainLayout->addWidget(outputGroup);
    mainLayout->addWidget(chartsGroup);
}

void MainWindow::setupMenuBar()
{
    menuBar = this->menuBar();
    
    // File Menu
    QMenu *fileMenu = menuBar->addMenu("&File");
    QAction *openDatasetAction = fileMenu->addAction("&Open Dataset...");
    QAction *saveModelAction = fileMenu->addAction("&Save Model...");
    QAction *loadModelAction = fileMenu->addAction("&Load Model...");
    fileMenu->addSeparator();
    QAction *exitAction = fileMenu->addAction("E&xit");
    
    // Training Menu
    QMenu *trainingMenu = menuBar->addMenu("&Training");
    QAction *startTrainingAction = trainingMenu->addAction("&Start Training");
    QAction *stopTrainingAction = trainingMenu->addAction("S&top Training");
    
    // Help Menu
    QMenu *helpMenu = menuBar->addMenu("&Help");
    QAction *aboutAction = helpMenu->addAction("&About");
    QAction *helpAction = helpMenu->addAction("&Help");
    
    // Connect menu actions
    connect(openDatasetAction, &QAction::triggered, this, &MainWindow::browseDataset);
    connect(saveModelAction, &QAction::triggered, this, &MainWindow::saveModel);
    connect(loadModelAction, &QAction::triggered, this, &MainWindow::loadModel);
    connect(exitAction, &QAction::triggered, this, &QWidget::close);
    connect(startTrainingAction, &QAction::triggered, this, &MainWindow::startTraining);
    connect(stopTrainingAction, &QAction::triggered, this, &MainWindow::stopTraining);
    connect(aboutAction, &QAction::triggered, this, &MainWindow::showAbout);
    connect(helpAction, &QAction::triggered, this, &MainWindow::showHelp);
}

void MainWindow::setupStatusBar()
{
    statusBar = this->statusBar();
    statusBar->showMessage("Ready");
}

void MainWindow::connectSignals()
{
    // Button connections
    connect(browseDatasetBtn, &QPushButton::clicked, this, &MainWindow::browseDataset);
    connect(browseOutputBtn, &QPushButton::clicked, this, &MainWindow::browseOutputPath);
    connect(startTrainingBtn, &QPushButton::clicked, this, &MainWindow::startTraining);
    connect(stopTrainingBtn, &QPushButton::clicked, this, &MainWindow::stopTraining);
    connect(saveModelBtn, &QPushButton::clicked, this, &MainWindow::saveModel);
    connect(loadModelBtn, &QPushButton::clicked, this, &MainWindow::loadModel);
    
    // Dataset manager connections
    connect(datasetManager, &DatasetManager::datasetLoaded, this, &MainWindow::updateDatasetInfo);
    
    // Model trainer connections
    connect(modelTrainer, &ModelTrainer::progressUpdated, this, &MainWindow::updateProgress);
    connect(modelTrainer, &ModelTrainer::trainingFinished, this, &MainWindow::trainingFinished);
    
    // Thread connections
    connect(trainingThread, &QThread::started, modelTrainer, &ModelTrainer::startTraining);
    connect(modelTrainer, &ModelTrainer::trainingFinished, trainingThread, &QThread::quit);
    
    // Line edit connections
    connect(datasetPathEdit, &QLineEdit::textChanged, [this](const QString &path) {
        if (!path.isEmpty()) {
            datasetManager->loadDataset(path);
        }
    });
}

void MainWindow::browseDataset()
{
    QString path = QFileDialog::getExistingDirectory(this, "Select Dataset Directory");
    if (!path.isEmpty()) {
        datasetPathEdit->setText(path);
        datasetManager->loadDataset(path);
    }
}

void MainWindow::browseOutputPath()
{
    QString path = QFileDialog::getExistingDirectory(this, "Select Output Directory");
    if (!path.isEmpty()) {
        outputPathEdit->setText(path);
    }
}

void MainWindow::startTraining()
{
    if (datasetPathEdit->text().isEmpty()) {
        QMessageBox::warning(this, "Warning", "Please select a dataset first.");
        return;
    }
    
    if (outputPathEdit->text().isEmpty()) {
        QMessageBox::warning(this, "Warning", "Please select an output directory.");
        return;
    }
    
    // Configure model trainer
    modelTrainer->setDatasetPath(datasetPathEdit->text());
    modelTrainer->setOutputPath(outputPathEdit->text());
    modelTrainer->setModelType(modelTypeCombo->currentText());
    modelTrainer->setInputSize(inputSizeSpin->value());
    modelTrainer->setHiddenSize(hiddenSizeSpin->value());
    modelTrainer->setOutputSize(outputSizeSpin->value());
    modelTrainer->setLearningRate(learningRateSpin->value());
    modelTrainer->setBatchSize(batchSizeSpin->value());
    modelTrainer->setEpochs(epochsSpin->value());
    
    // Clear previous training data
    lossHistory.clear();
    accuracyHistory.clear();
    epochHistory.clear();
    lossChart->clearData();
    accuracyChart->clearData();
    
    // Start training
    trainingThread->start();
    updateTrainingStatus(true);
    
    logMessage("Training started with " + QString::number(epochsSpin->value()) + " epochs");
}

void MainWindow::stopTraining()
{
    if (isTraining) {
        modelTrainer->stopTraining();
        updateTrainingStatus(false);
        logMessage("Training stopped by user");
    }
}

void MainWindow::saveModel()
{
    QString path = QFileDialog::getSaveFileName(this, "Save Model", "", "Model Files (*.model);;All Files (*)");
    if (!path.isEmpty()) {
        if (modelTrainer->saveModel(path)) {
            logMessage("Model saved successfully to: " + path);
        } else {
            logMessage("Failed to save model");
        }
    }
}

void MainWindow::loadModel()
{
    QString path = QFileDialog::getOpenFileName(this, "Load Model", "", "Model Files (*.model);;All Files (*)");
    if (!path.isEmpty()) {
        if (modelTrainer->loadModel(path)) {
            logMessage("Model loaded successfully from: " + path);
        } else {
            logMessage("Failed to load model");
        }
    }
}

void MainWindow::updateProgress(int epoch, double loss, double accuracy)
{
    // Update progress bar
    int progress = (epoch * 100) / epochsSpin->value();
    trainingProgressBar->setValue(progress);
    
    // Update labels
    currentEpochLabel->setText(QString("Epoch: %1/%2").arg(epoch).arg(epochsSpin->value()));
    currentLossLabel->setText(QString("Loss: %1").arg(loss, 0, 'f', 4));
    currentAccuracyLabel->setText(QString("Accuracy: %1%").arg(accuracy * 100, 0, 'f', 2));
    
    // Store data for charts
    epochHistory.append(epoch);
    lossHistory.append(loss);
    accuracyHistory.append(accuracy);
    
    // Update charts
    lossChart->addDataPoint(epoch, loss);
    accuracyChart->addDataPoint(epoch, accuracy * 100);
    
    // Update status bar
    statusBar->showMessage(QString("Training - Epoch %1/%2, Loss: %3, Accuracy: %4%")
                          .arg(epoch).arg(epochsSpin->value()).arg(loss, 0, 'f', 4).arg(accuracy * 100, 0, 'f', 2));
}

void MainWindow::trainingFinished()
{
    updateTrainingStatus(false);
    trainingProgressBar->setValue(100);
    logMessage("Training completed successfully");
    statusBar->showMessage("Training completed");
    
    QMessageBox::information(this, "Training Complete", "Model training has been completed successfully!");
}

void MainWindow::showAbout()
{
    QMessageBox::about(this, "About Model Training Application",
                      "Model Training Application v1.0.0\n\n"
                      "A comprehensive GUI application for training machine learning models.\n\n"
                      "Features:\n"
                      "- Support for multiple model types\n"
                      "- Real-time training progress visualization\n"
                      "- Dataset management and preview\n"
                      "- Model save/load functionality\n"
                      "- Detailed logging and metrics\n\n"
                      "Built with Qt6 and C++");
}

void MainWindow::showHelp()
{
    QMessageBox::information(this, "Help",
                           "How to use the Model Training Application:\n\n"
                           "1. Select a dataset using the Browse button\n"
                           "2. Configure your model parameters\n"
                           "3. Choose an output directory\n"
                           "4. Click 'Start Training' to begin\n"
                           "5. Monitor progress in real-time\n"
                           "6. Save your trained model when complete\n\n"
                           "For more detailed information, please refer to the documentation.");
}

void MainWindow::updateDatasetInfo()
{
    // This would be implemented to show dataset statistics
    datasetInfoLabel->setText("Dataset loaded successfully");
    datasetInfoLabel->setStyleSheet("color: green; font-weight: bold;");
    
    // Update preview table with sample data
    // This is a placeholder - actual implementation would load real data
    datasetPreviewTable->setRowCount(5);
    for (int i = 0; i < 5; ++i) {
        datasetPreviewTable->setItem(i, 0, new QTableWidgetItem(QString("Sample %1").arg(i + 1)));
        datasetPreviewTable->setItem(i, 1, new QTableWidgetItem("784 features"));
        datasetPreviewTable->setItem(i, 2, new QTableWidgetItem(QString::number(i % 10)));
        datasetPreviewTable->setItem(i, 3, new QTableWidgetItem("Training"));
        datasetPreviewTable->setItem(i, 4, new QTableWidgetItem("Valid"));
    }
}

void MainWindow::updateTrainingStatus(bool training)
{
    isTraining = training;
    startTrainingBtn->setEnabled(!training);
    stopTrainingBtn->setEnabled(training);
    saveModelBtn->setEnabled(!training);
    
    if (training) {
        statusBar->showMessage("Training in progress...");
    } else {
        statusBar->showMessage("Ready");
    }
}

void MainWindow::logMessage(const QString &message)
{
    QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss");
    logTextEdit->append(QString("[%1] %2").arg(timestamp).arg(message));
} 