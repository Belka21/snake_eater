#pragma execution_character_set("utf-8")
#include "QtWidgetsApplication1.h"
#include "ui_QtWidgetsApplication1.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QTimer>
#include <QTextCodec> 

using namespace cv;

const int MIN_GOOD_MATCHES = 8;          // Минимальное количество совпадений
const float MIN_MATCH_RATIO = 0.1f;      // 15% минимального совпадения
const float GOOD_MATCH_THRESHOLD = 0.7f;  // Порог для соотношения расстояний

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->resultLabel->setText("Результат появится здесь");

    // Настройка интерфейса
    ui->preprocessingLevel->setRange(1, 5);
    ui->preprocessingLevel->setValue(3);
    ui->progressBar->setVisible(false);

    // Загрузка базы данных
    if (!database.load()) {
        QMessageBox::warning(this,
            QString::fromUtf8(u8"Ошибка"),
            QString::fromUtf8(u8"Не удалось загрузить базу данных!"));
    }

    clearResults();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_selectImageButton_clicked()
{
    QString filePath = QFileDialog::getOpenFileName(
        this, "Выберите фото змеи", "",
        "Images (*.jpg *.jpeg *.png)");

    if (!filePath.isEmpty()) {
        currentImage = imread(filePath.toStdString());
        if (!currentImage.empty()) {
            displayImage(currentImage, ui->originalImageLabel);
            clearResults();
        }
        else {
            QMessageBox::warning(this, QString::fromUtf8(u8"Ошибка"), "Не удалось загрузить изображение!");
        }
    }
}

void drawMatches(const cv::Mat& img1, const std::vector<cv::KeyPoint>& kp1,
    const cv::Mat& img2, const std::vector<cv::KeyPoint>& kp2,
    const std::vector<cv::DMatch>& matches)
{
    cv::Mat img_matches;
    cv::drawMatches(img1, kp1, img2, kp2, matches, img_matches);
    cv::imshow("Matches", img_matches);
    cv::waitKey(0);
}

void MainWindow::on_processButton_clicked()
{
    if (currentImage.empty()) {
        QMessageBox::warning(this, QString::fromUtf8(u8"Ошибка"), "Сначала выберите изображение!");
        return;
    }

    enableControls(false);
    ui->progressBar->setVisible(true);
    ui->progressBar->setValue(0);

    // Имитация прогресса (в реальном приложении замените на фактический прогресс)
    QTimer::singleShot(100, [this]() { ui->progressBar->setValue(20); });

    // Предобработка
    bool showSteps = ui->showStepsCheckBox->isChecked();
    int level = ui->preprocessingLevel->value();

    QTimer::singleShot(200, [this, showSteps, level]() {
        if (showSteps) {
            ImagePreprocessor::showProcessingSteps(currentImage);
        }

        processedImage = ImagePreprocessor::preprocess(
            currentImage, true, true, level);

        displayImage(processedImage, ui->processedImageLabel);
        ui->progressBar->setValue(40);

        // Извлечение ключевых точек
        currentFeatures = detectSIFTFeatures(processedImage);
        ui->progressBar->setValue(60);

        // Поиск в базе данных
        std::vector<std::string> allSnakes = database.getAllSnakeNames();
        float bestMatchScore = 0;
        matchedSnakeName.clear();

        if (!allSnakes.empty()) {
            for (const auto& name : allSnakes) {
                SnakeFeatures dbFeatures = database.getSnakeFeatures(name);

                // Пропускаем пустые записи
                if (dbFeatures.keypoints.empty() || dbFeatures.descriptors.empty()) {
                    continue;
                }

                MatchResult match = matchFeatures(currentFeatures,
                    { dbFeatures.keypoints, dbFeatures.descriptors },
                    cv::NORM_L2, GOOD_MATCH_THRESHOLD);

                // Рассчитываем процент совпадений относительно меньшего изображения
                size_t minFeatures = std::min(
                    currentFeatures.keypoints.size(),
                    dbFeatures.keypoints.size()
                );

                float matchRatio = minFeatures > 0 ?
                    (float)match.good_matches / minFeatures : 0;

                //qDebug() << "Match with" << QString::fromStdString(name)
                //    << ":" << match.good_matches << "matches,"
                //    << "ratio:" << matchRatio;

                // Критерии принятия решения
                if (match.good_matches >= MIN_GOOD_MATCHES &&
                    matchRatio >= MIN_MATCH_RATIO &&
                    matchRatio > bestMatchScore)
                {
                    bestMatchScore = matchRatio;
                    matchedSnakeName = name;
                    matchedSnakeImage = cv::imread(dbFeatures.image_paths[0]);
                    //drawMatches();
                }
            }
        }

        // Отображение результатов
        if (!matchedSnakeName.empty()) {
            ui->resultLabel->setText(
                QString("Совпадение найдено: %1\nСовпадений: %2 (%3%)")
                .arg(QString::fromStdString(matchedSnakeName))
                .arg(bestMatchScore)
                .arg(static_cast<int>(100 * bestMatchScore /
                    std::min(currentFeatures.keypoints.size(),
                        database.getSnakeFeatures(matchedSnakeName).keypoints.size()))));
            displayImage(matchedSnakeImage, ui->matchedImageLabel);
        }
        else {
            ui->resultLabel->setText("Совпадений не найдено\n(недостаточно хороших совпадений)");
            ui->saveGroupBox->setEnabled(true);
        }

        ui->progressBar->setValue(100);
        QTimer::singleShot(500, [this]() {
            ui->progressBar->setVisible(false);
            enableControls(true);
            });
        });
}

void MainWindow::on_saveToDbButton_clicked()
{
    QString name = ui->snakeNameEdit->text().trimmed();
    if (name.isEmpty()) {
        QMessageBox::warning(this, "Ошибка", "Введите имя змеи!");
        return;
    }

    if (database.addSnake(name.toStdString(),
        currentFeatures.keypoints,
        currentFeatures.descriptors,
        currentImage)) {
        database.save();
        QMessageBox::information(this, "Успех", "Змея добавлена в базу данных!");
        ui->saveGroupBox->setEnabled(false);
    }
    else {
        QMessageBox::warning(this, "Ошибка", "Не удалось добавить змею в базу!");
    }
}

void MainWindow::displayImage(const cv::Mat& mat, QLabel* label)
{
    if (mat.empty() || !label) {
        //qDebug() << "Invalid input: empty image or null label";
        label->clear();
        return;
    }

    try {
        // Создаем глубокую копию данных OpenCV
        cv::Mat matCopy = mat.clone();

        // Конвертируем в QImage с явным копированием данных
        QImage img;
        if (matCopy.channels() == 1) {
            img = QImage(matCopy.data, matCopy.cols, matCopy.rows,
                static_cast<int>(matCopy.step),
                QImage::Format_Grayscale8).copy();
        }
        else {
            cv::Mat rgb;
            cv::cvtColor(matCopy, rgb, cv::COLOR_BGR2RGB);
            img = QImage(rgb.data, rgb.cols, rgb.rows,
                static_cast<int>(rgb.step),
                QImage::Format_RGB888).copy();
        }

        // Проверяем, что изображение создано успешно
        if (img.isNull()) {
            //qDebug() << "Failed to create QImage from OpenCV data";
            label->clear();
            return;
        }

        // Масштабируем и отображаем
        QPixmap pixmap = QPixmap::fromImage(img);
        if (pixmap.isNull()) {
            //qDebug() << "Failed to create QPixmap from QImage";
            label->clear();
            return;
        }

        label->setPixmap(pixmap.scaled(
            label->width(),
            label->height(),
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation
        ));
    }
    catch (const cv::Exception& e) {
       // qDebug() << "OpenCV exception:" << e.what();
        label->clear();
    }
    catch (const std::exception& e) {
        //qDebug() << "Standard exception:" << e.what();
        label->clear();
    }
    catch (...) {
        //qDebug() << "Unknown exception in displayImage";
        label->clear();
    }
}

void MainWindow::clearResults()
{
    ui->processedImageLabel->clear();
    ui->matchedImageLabel->clear();
    ui->resultLabel->setText("Результат появится здесь");
    ui->saveGroupBox->setEnabled(false);
    ui->snakeNameEdit->clear();
    matchedSnakeName.clear();
}

void MainWindow::enableControls(bool enable)
{
    ui->selectImageButton->setEnabled(enable);
    ui->processButton->setEnabled(enable);
    ui->preprocessingLevel->setEnabled(enable);
    ui->showStepsCheckBox->setEnabled(enable);
}