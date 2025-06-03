#include "image_preprocessing.h"
#include <opencv2/ximgproc.hpp>

using namespace cv;

Mat ImagePreprocessor::preprocess(const Mat& input,
    bool enhance_scales,
    bool remove_background,
    int scale_enhancement_level) {
    if (input.empty()) return input;

    Mat processed = input.clone();

    // Конвертируем в grayscale если нужно
    if (processed.channels() > 1) {
        cvtColor(processed, processed, COLOR_BGR2GRAY);
    }

    // Нормализация освещения
    processed = normalizeLighting(processed);

    // Удаление фона (если требуется)
    if (remove_background) {
        processed = removeBackground(processed);
    }

    // Улучшение контраста
    processed = enhanceContrast(processed);

    // Удаление шума
    processed = removeNoise(processed);

    // Специальное улучшение чешуи
    if (enhance_scales) {
        processed = enhanceScales(processed, scale_enhancement_level);
    }

    // Финалное повышение резкости
    processed = sharpenImage(processed);

    return processed;
}

Mat ImagePreprocessor::removeNoise(const Mat& input) {
    Mat denoised;

    // Нелинейное подавление шума (хорошо для сохранения границ)
    fastNlMeansDenoising(input, denoised,
        10,  // h: параметр силы фильтрации
        7,   // templateWindowSize
        21); // searchWindowSize

    // Альтернативный вариант - билатеральный фильтр
    // bilateralFilter(input, denoised, 9, 75, 75);

    return denoised;
}

Mat ImagePreprocessor::enhanceContrast(const Mat& input) {
    Mat enhanced;

    // CLAHE (Contrast Limited Adaptive Histogram Equalization)
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(4);
    clahe->apply(input, enhanced);

    return enhanced;
}

Mat ImagePreprocessor::normalizeLighting(const Mat& input) {
    Mat normalized;

    // Вычитание размытой версии для нормализации освещения
    Mat blurred;
    GaussianBlur(input, blurred, Size(101, 101), 0);
    normalized = input - blurred + 128; // 128 добавляем для среднего значения

    return normalized;
}

Mat ImagePreprocessor::enhanceScales(const Mat& input, int level) {
    Mat scales_enhanced;

    // Используем фильтр разностки Гауссианов (DoG) для выделения чешуи
    Mat gauss1, gauss2;
    GaussianBlur(input, gauss1, Size(0, 0), 1.0 * level);
    GaussianBlur(input, gauss2, Size(0, 0), 2.0 * level);
    subtract(gauss1, gauss2, scales_enhanced);

    // Усиливаем контраст
    normalize(scales_enhanced, scales_enhanced, 0, 255, NORM_MINMAX);

    // Комбинируем с оригиналом
    addWeighted(input, 0.7, scales_enhanced, 0.3, 0, scales_enhanced);

    return scales_enhanced;
}

Mat ImagePreprocessor::removeBackground(const Mat& input) {
    Mat foreground;

    // Используем адаптивный порог
    adaptiveThreshold(input, foreground, 255,
        ADAPTIVE_THRESH_GAUSSIAN_C,
        THRESH_BINARY_INV,
        51,  // blockSize
        10); // C

    // Улучшаем маску морфологическими операциями
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(foreground, foreground, MORPH_CLOSE, kernel);
    morphologyEx(foreground, foreground, MORPH_OPEN, kernel);

    // Применяем маску
    Mat result;
    input.copyTo(result, foreground);

    return result;
}

Mat ImagePreprocessor::sharpenImage(const Mat& input) {
    Mat sharpened;

    // Ядро для повышения резкости
    Mat kernel = (Mat_<float>(3, 3) <<
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0);

    filter2D(input, sharpened, input.depth(), kernel);

    return sharpened;
}

void ImagePreprocessor::showProcessingSteps(const cv::Mat& input, const std::string& window_name) {
    // Создаем большое изображение для отображения всех этапов
    cv::Mat canvas;
    std::vector<cv::Mat> steps;

    // Получаем изображения после каждого этапа обработки
    cv::Mat after_lighting = normalizeLighting(input.clone());
    cv::Mat after_denoise = removeNoise(after_lighting.clone());
    cv::Mat after_contrast = enhanceContrast(after_denoise.clone());
    cv::Mat after_scales = enhanceScales(after_contrast.clone(), 2);
    cv::Mat after_sharpen = sharpenImage(after_scales.clone());
    cv::Mat final = preprocess(input.clone());

    // Подписываем этапы
    auto addLabel = [](cv::Mat& img, const std::string& label) {
        cv::putText(img, label, cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.8,
            cv::Scalar(255, 255, 255), 2);
        };

    addLabel(after_lighting, "1. Lighting Normalization");
    addLabel(after_denoise, "2. Noise Removal");
    addLabel(after_contrast, "3. Contrast Enhancement");
    addLabel(after_scales, "4. Scales Enhancement");
    addLabel(after_sharpen, "5. Sharpening");
    addLabel(final, "6. Final Result");

    // Собираем все этапы в один коллаж
    cv::hconcat(std::vector<cv::Mat>{after_lighting, after_denoise, after_contrast}, canvas);
    cv::Mat bottom_row;
    cv::hconcat(std::vector<cv::Mat>{after_scales, after_sharpen, final}, bottom_row);
    cv::vconcat(canvas, bottom_row, canvas);

    // Показываем результат
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1200, 800);
    cv::imshow(window_name, canvas);
    cv::waitKey(0);
    cv::destroyWindow(window_name);
}