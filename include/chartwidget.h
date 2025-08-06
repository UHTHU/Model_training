#ifndef CHARTWIDGET_H
#define CHARTWIDGET_H

#include <QWidget>
#include <QVBoxLayout>
#include <QChartView>
#include <QChart>
#include <QSplineSeries>
#include <QValueAxis>
#include <QCategoryAxis>
#include <QVector>

QT_CHARTS_USE_NAMESPACE

class ChartWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ChartWidget(const QString &title, QWidget *parent = nullptr);
    ~ChartWidget();

    void addDataPoint(double x, double y);
    void addDataPoints(const QVector<double> &xValues, const QVector<double> &yValues);
    void clearData();
    void setYAxisRange(double min, double max);
    void setXAxisRange(double min, double max);
    void setTitle(const QString &title);

private:
    void setupChart();
    void updateAxis();

    QVBoxLayout *layout;
    QChartView *chartView;
    QChart *chart;
    QSplineSeries *series;
    QValueAxis *xAxis;
    QValueAxis *yAxis;
    
    QVector<double> xData;
    QVector<double> yData;
    double yMin, yMax;
    double xMin, xMax;
    bool autoRange;
};

#endif // CHARTWIDGET_H 