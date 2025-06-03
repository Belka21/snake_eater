#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QImage>
#include <QPixmap>
#include "image_preprocessing.h"
#include "snake_database.h"
#include "image_comparison.h"
#include <qlabel.h>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

private slots:
    void on_selectImageButton_clicked();
    void on_processButton_clicked();
    void on_saveToDbButton_clicked();

private:
    Ui::MainWindow* ui;
    SnakeDatabase database;
    cv::Mat currentImage;
    cv::Mat processedImage;
    FeaturePoints currentFeatures;
    std::string matchedSnakeName;
    cv::Mat matchedSnakeImage;

    void displayImage(const cv::Mat& mat, QLabel* label);
    void clearResults();
    void enableControls(bool enable);
};
#endif // MAINWINDOW_H