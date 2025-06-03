#pragma once
#ifndef IMAGE_PREPROCESSING_H
#define IMAGE_PREPROCESSING_H

#include <opencv2/opencv.hpp>
#include <string>

class ImagePreprocessor {
public:
    // Основной метод для предварительной обработки
    static cv::Mat preprocess(const cv::Mat& input,
        bool enhance_scales = true,
        bool remove_background = false,
        int scale_enhancement_level = 2);

    static void showProcessingSteps(const cv::Mat& input, 
                const std::string& window_name = "Processing Steps");

private:
    // Методы для конкретных этапов обработки
    static cv::Mat removeNoise(const cv::Mat& input);
    static cv::Mat enhanceContrast(const cv::Mat& input);
    static cv::Mat normalizeLighting(const cv::Mat& input);
    static cv::Mat enhanceScales(const cv::Mat& input, int level);
    static cv::Mat removeBackground(const cv::Mat& input);
    static cv::Mat sharpenImage(const cv::Mat& input);
};

#endif // IMAGE_PREPROCESSING_H