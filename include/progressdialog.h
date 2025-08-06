#ifndef PROGRESSDIALOG_H
#define PROGRESSDIALOG_H

#include <QDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QTextEdit>
#include <QTimer>

class ProgressDialog : public QDialog
{
    Q_OBJECT

public:
    explicit ProgressDialog(QWidget *parent = nullptr);
    ~ProgressDialog();

    void setProgress(int value);
    void setStatus(const QString &status);
    void addLogMessage(const QString &message);
    void setCancelEnabled(bool enabled);
    bool wasCanceled() const;

signals:
    void canceled();

private slots:
    void onCancelClicked();

private:
    void setupUI();

    QVBoxLayout *mainLayout;
    QLabel *statusLabel;
    QProgressBar *progressBar;
    QTextEdit *logTextEdit;
    QPushButton *cancelButton;
    QTimer *updateTimer;
    
    bool canceledFlag;
    QString currentStatus;
    QStringList logMessages;
};

#endif // PROGRESSDIALOG_H 