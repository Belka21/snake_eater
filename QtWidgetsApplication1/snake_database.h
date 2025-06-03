#ifndef SNAKE_DATABASE_H
#define SNAKE_DATABASE_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct SnakeFeatures {
    std::string name;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<std::string> image_paths;
};

class SnakeDatabase {
public:
    SnakeDatabase(const std::string& db_path = "snake_database");

    // Основные операции
    bool addSnake(const std::string& name,
        const std::vector<cv::KeyPoint>& keypoints,
        const cv::Mat& descriptors,
        const cv::Mat& image);

    bool removeSnake(const std::string& name);
    bool updateSnake(const std::string& name,
        const std::vector<cv::KeyPoint>& new_keypoints,
        const cv::Mat& new_descriptors,
        const cv::Mat& new_image);

    // Поиск
    bool findSnake(const cv::Mat& query_descriptors,
        std::string& found_name,
        double min_match_ratio = 0.3) const;

    // Получение данных
    std::vector<std::string> getAllSnakeNames() const;
    SnakeFeatures getSnakeFeatures(const std::string& name) const;
    cv::Mat getSnakeImage(const std::string& name, int index = 0) const;

    // Работа с базой
    bool save() const;
    bool load();
    bool exportTo(const std::string& file_path) const;
    bool importFrom(const std::string& file_path);

    // Статистика
    size_t count() const;
    std::map<std::string, size_t> getStatistics() const;

private:
    std::string db_path_;
    std::map<std::string, SnakeFeatures> snakes_;

    // Вспомогательные методы
    std::string generateImagePath(const std::string& snake_name, int index) const;
    json featuresToJson(const SnakeFeatures& features) const;
    SnakeFeatures jsonToFeatures(const json& j) const;
    bool saveImage(const cv::Mat& image, const std::string& path) const;
};

#endif // SNAKE_DATABASE_H