#include "progressdialog.h"
#include <QApplication>
#include <QDateTime>

ProgressDialog::ProgressDialog(QWidget *parent)
    : QDialog(parent)
    , canceledFlag(false)
{
    setupUI();
    
    // Set window properties
    setWindowTitle("Training Progress");
    setModal(true);
    setFixedSize(500, 400);
    
    // Center on parent
    if (parent) {
        QPoint parentCenter = parent->geometry().center();
        QPoint dialogCenter = geometry().center();
        move(parentCenter - dialogCenter);
    }
}

ProgressDialog::~ProgressDialog()
{
}

void ProgressDialog::setProgress(int value)
{
    progressBar->setValue(value);
}

void ProgressDialog::setStatus(const QString &status)
{
    currentStatus = status;
    statusLabel->setText(status);
}

void ProgressDialog::addLogMessage(const QString &message)
{
    QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss");
    QString logEntry = QString("[%1] %2").arg(timestamp).arg(message);
    logMessages.append(logEntry);
    
    // Keep only last 100 messages
    if (logMessages.size() > 100) {
        logMessages.removeFirst();
    }
    
    // Update log display
    logTextEdit->clear();
    logTextEdit->setPlainText(logMessages.join("\n"));
    
    // Scroll to bottom
    QTextCursor cursor = logTextEdit->textCursor();
    cursor.movePosition(QTextCursor::End);
    logTextEdit->setTextCursor(cursor);
}

void ProgressDialog::setCancelEnabled(bool enabled)
{
    cancelButton->setEnabled(enabled);
}

bool ProgressDialog::wasCanceled() const
{
    return canceledFlag;
}

void ProgressDialog::onCancelClicked()
{
    canceledFlag = true;
    cancelButton->setEnabled(false);
    cancelButton->setText("Canceling...");
    emit canceled();
}

void ProgressDialog::setupUI()
{
    mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(10);
    mainLayout->setContentsMargins(20, 20, 20, 20);
    
    // Status label
    statusLabel = new QLabel("Initializing training...", this);
    statusLabel->setStyleSheet("font-weight: bold; font-size: 12px;");
    mainLayout->addWidget(statusLabel);
    
    // Progress bar
    progressBar = new QProgressBar(this);
    progressBar->setRange(0, 100);
    progressBar->setValue(0);
    progressBar->setTextVisible(true);
    progressBar->setFormat("Progress: %p%");
    mainLayout->addWidget(progressBar);
    
    // Log text area
    QLabel *logLabel = new QLabel("Training Log:", this);
    logLabel->setStyleSheet("font-weight: bold;");
    mainLayout->addWidget(logLabel);
    
    logTextEdit = new QTextEdit(this);
    logTextEdit->setReadOnly(true);
    logTextEdit->setFont(QFont("Consolas", 9));
    logTextEdit->setMaximumHeight(200);
    mainLayout->addWidget(logTextEdit);
    
    // Cancel button
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    buttonLayout->addStretch();
    
    cancelButton = new QPushButton("Cancel Training", this);
    cancelButton->setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 8px 16px; }");
    connect(cancelButton, &QPushButton::clicked, this, &ProgressDialog::onCancelClicked);
    
    buttonLayout->addWidget(cancelButton);
    mainLayout->addLayout(buttonLayout);
    
    // Add initial log message
    addLogMessage("Progress dialog initialized");
} 