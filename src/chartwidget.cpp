#include "chartwidget.h"
#include <QApplication>
#include <QScreen>
#include <algorithm>

ChartWidget::ChartWidget(const QString &title, QWidget *parent)
    : QWidget(parent)
    , yMin(0.0)
    , yMax(1.0)
    , xMin(0.0)
    , xMax(100.0)
    , autoRange(true)
{
    setupChart();
    setTitle(title);
    
    // Set minimum size
    setMinimumSize(300, 200);
}

ChartWidget::~ChartWidget()
{
}

void ChartWidget::setupChart()
{
    layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    
    // Create chart
    chart = new QChart();
    chart->setAnimationOptions(QChart::SeriesAnimations);
    chart->setMargins(QMargins(10, 10, 10, 10));
    
    // Create series
    series = new QSplineSeries();
    series->setName("Training Progress");
    series->setUseOpenGL(true); // Enable hardware acceleration
    
    // Create axes
    xAxis = new QValueAxis();
    yAxis = new QValueAxis();
    
    // Configure axes
    xAxis->setTitleText("Epoch");
    xAxis->setLabelFormat("%.0f");
    xAxis->setTickCount(11);
    xAxis->setRange(xMin, xMax);
    
    yAxis->setTitleText("Value");
    yAxis->setLabelFormat("%.4f");
    yAxis->setTickCount(11);
    yAxis->setRange(yMin, yMax);
    
    // Add series and axes to chart
    chart->addSeries(series);
    chart->setAxisX(xAxis, series);
    chart->setAxisY(yAxis, series);
    
    // Create chart view
    chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);
    chartView->setRubberBand(QChartView::RectangleRubberBand);
    chartView->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    chartView->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    
    layout->addWidget(chartView);
}

void ChartWidget::addDataPoint(double x, double y)
{
    xData.append(x);
    yData.append(y);
    
    // Add point to series
    series->append(x, y);
    
    // Update axis ranges if auto-range is enabled
    if (autoRange) {
        updateAxis();
    }
    
    // Keep only last 1000 points for performance
    if (xData.size() > 1000) {
        xData.removeFirst();
        yData.removeFirst();
        
        // Rebuild series with limited data
        series->clear();
        for (int i = 0; i < xData.size(); ++i) {
            series->append(xData[i], yData[i]);
        }
    }
}

void ChartWidget::addDataPoints(const QVector<double> &xValues, const QVector<double> &yValues)
{
    if (xValues.size() != yValues.size()) {
        return;
    }
    
    for (int i = 0; i < xValues.size(); ++i) {
        addDataPoint(xValues[i], yValues[i]);
    }
}

void ChartWidget::clearData()
{
    xData.clear();
    yData.clear();
    series->clear();
    
    if (autoRange) {
        xMin = 0.0;
        xMax = 100.0;
        yMin = 0.0;
        yMax = 1.0;
        updateAxis();
    }
}

void ChartWidget::setYAxisRange(double min, double max)
{
    yMin = min;
    yMax = max;
    autoRange = false;
    yAxis->setRange(yMin, yMax);
}

void ChartWidget::setXAxisRange(double min, double max)
{
    xMin = min;
    xMax = max;
    autoRange = false;
    xAxis->setRange(xMin, xMax);
}

void ChartWidget::setTitle(const QString &title)
{
    chart->setTitle(title);
    chart->setTitleFont(QFont("Arial", 12, QFont::Bold));
}

void ChartWidget::updateAxis()
{
    if (xData.isEmpty() || yData.isEmpty()) {
        return;
    }
    
    // Calculate ranges
    double newXMin = *std::min_element(xData.begin(), xData.end());
    double newXMax = *std::max_element(xData.begin(), xData.end());
    double newYMin = *std::min_element(yData.begin(), yData.end());
    double newYMax = *std::max_element(yData.begin(), yData.end());
    
    // Add some padding
    double xPadding = (newXMax - newXMin) * 0.05;
    double yPadding = (newYMax - newYMin) * 0.1;
    
    if (xPadding < 1.0) xPadding = 1.0;
    if (yPadding < 0.01) yPadding = 0.01;
    
    newXMin -= xPadding;
    newXMax += xPadding;
    newYMin -= yPadding;
    newYMax += yPadding;
    
    // Ensure minimum ranges
    if (newXMax - newXMin < 10.0) {
        double center = (newXMin + newXMax) / 2.0;
        newXMin = center - 5.0;
        newXMax = center + 5.0;
    }
    
    if (newYMax - newYMin < 0.1) {
        double center = (newYMin + newYMax) / 2.0;
        newYMin = center - 0.05;
        newYMax = center + 0.05;
    }
    
    // Update axis ranges
    xAxis->setRange(newXMin, newXMax);
    yAxis->setRange(newYMin, newYMax);
} 