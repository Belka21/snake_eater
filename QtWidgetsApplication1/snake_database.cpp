#pragma execution_character_set("utf-8")
#include "snake_database.h"
#include <filesystem>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <windows.h>

namespace fs = std::filesystem;

SnakeDatabase::SnakeDatabase(const std::string& db_path) : db_path_(db_path) {
    fs::create_directories(db_path_ + "/data/points");
    fs::create_directories(db_path_ + "/data/images");
}

std::string utf8_to_cp1251(const std::string& utf8_str) {
    if (utf8_str.empty()) return {};

    // Конвертация UTF-8 → UTF-16 (Windows Wide-String)
    int wide_len = MultiByteToWideChar(CP_UTF8, 0, utf8_str.c_str(), -1, nullptr, 0);
    if (wide_len == 0) return {};

    std::wstring utf16_str(wide_len, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8_str.c_str(), -1, &utf16_str[0], wide_len);

    // Конвертация UTF-16 → CP1251
    int cp1251_len = WideCharToMultiByte(1251, 0, utf16_str.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (cp1251_len == 0) return {};

    std::string cp1251_str(cp1251_len, '\0');
    WideCharToMultiByte(1251, 0, utf16_str.c_str(), -1, &cp1251_str[0], cp1251_len, nullptr, nullptr);

    return cp1251_str;
}

bool SnakeDatabase::addSnake(const std::string& name,
    const std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& descriptors,
    const cv::Mat& image) {
    if (name.empty() || keypoints.empty() || descriptors.empty() || image.empty()) {
        return false;
    }

    // Проверяем, есть ли уже такая змея
    if (snakes_.find(name) != snakes_.end()) {
        return false;
    }

    // Сохраняем изображение
    std::string img_dir = db_path_ + "/data/images/" + name;
	std::string img_dir_cp1251 = utf8_to_cp1251(img_dir);
    fs::create_directories(img_dir_cp1251);

    std::string img_path = img_dir + "/photo_0.jpg";
    std::string img_path_cp1251 = utf8_to_cp1251(img_path);
    if (!saveImage(image, img_path_cp1251)) {
        return false;
    }

    // Создаем запись
    SnakeFeatures features;
    features.name = name;
    features.keypoints = keypoints;
    features.descriptors = descriptors;
    features.image_paths.push_back(img_path);

    snakes_[name] = features;
    return true;
}

bool SnakeDatabase::removeSnake(const std::string& name) {
    auto it = snakes_.find(name);
    if (it == snakes_.end()) {
        return false;
    }

    // Удаляем связанные изображения
    for (const auto& img_path : it->second.image_paths) {
        fs::remove(img_path);
    }

    // Удаляем папку с изображениями
    std::string img_dir = db_path_ + "/data/images/" + name;
    if (fs::exists(img_dir)) {
        fs::remove_all(img_dir);
    }

    // Удаляем файл с ключевыми точками
    std::string points_path = db_path_ + "/data/points/" + name + ".json";
    fs::remove(points_path);

    // Удаляем из памяти
    snakes_.erase(it);
    return true;
}

bool SnakeDatabase::updateSnake(const std::string& name,
    const std::vector<cv::KeyPoint>& new_keypoints,
    const cv::Mat& new_descriptors,
    const cv::Mat& new_image) {
    auto it = snakes_.find(name);
    if (it == snakes_.end()) {
        return false;
    }

    // Обновляем ключевые точки и дескрипторы
    it->second.keypoints = new_keypoints;
    it->second.descriptors = new_descriptors;

    // Добавляем новое изображение
    std::string img_path = generateImagePath(name, it->second.image_paths.size());
    if (!saveImage(new_image, img_path)) {
        return false;
    }
    it->second.image_paths.push_back(img_path);

    return true;
}

bool SnakeDatabase::findSnake(const cv::Mat& query_descriptors,
    std::string& found_name,
    double min_match_ratio) const {
    if (query_descriptors.empty() || snakes_.empty()) {
        return false;
    }

    double best_match_score = 0;
    std::string best_match_name;

    for (const auto& [name, features] : snakes_) {
        if (features.descriptors.empty()) continue;

        // Сопоставление дескрипторов
        cv::BFMatcher matcher(cv::NORM_L2);
        std::vector<cv::DMatch> matches;
        matcher.match(query_descriptors, features.descriptors, matches);

        // Фильтрация хороших совпадений
        double min_dist = DBL_MAX;
        for (const auto& m : matches) {
            if (m.distance < min_dist) {
                min_dist = m.distance;
            }
        }

        int good_matches = 0;
        for (const auto& m : matches) {
            if (m.distance < 3 * min_dist) {
                good_matches++;
            }
        }

        double match_score = static_cast<double>(good_matches) / matches.size();
        if (match_score > best_match_score) {
            best_match_score = match_score;
            best_match_name = name;
        }
    }

    if (best_match_score >= min_match_ratio) {
        found_name = best_match_name;
        return true;
    }

    return false;
}

std::vector<std::string> SnakeDatabase::getAllSnakeNames() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : snakes_) {
        names.push_back(name);
    }
    return names;
}

SnakeFeatures SnakeDatabase::getSnakeFeatures(const std::string& name) const {
    auto it = snakes_.find(name);
    if (it != snakes_.end()) {
        return it->second;
    }
    return SnakeFeatures();
}

cv::Mat SnakeDatabase::getSnakeImage(const std::string& name, int index) const {
    auto it = snakes_.find(name);
    if (it != snakes_.end() && index < it->second.image_paths.size()) {
        return cv::imread(it->second.image_paths[index]);
    }
    return cv::Mat();
}

bool SnakeDatabase::save() const {
    json meta;
    for (const auto& [name, features] : snakes_) {
        // Сохраняем ключевые точки
        std::string points_path = db_path_ + "/data/points/" + name + ".json";
        std::ofstream points_file(points_path);
        if (!points_file.is_open()) {
            return false;
        }
        points_file << featuresToJson(features);

        // Добавляем в мета-информацию
        meta[name] = {
            {"points", points_path},
            {"images", features.image_paths}
        };
    }

    std::ofstream meta_file(db_path_ + "/meta.json");
    if (!meta_file.is_open()) {
        return false;
    }
    meta_file << meta.dump(4);

    return true;
}

bool SnakeDatabase::load() {
    std::ifstream meta_file(db_path_ + "/meta.json");
    if (!meta_file.is_open()) {
        return false;
    }

    json meta;
    try {
        meta_file >> meta;
    }
    catch (...) {
        return false;
    }

    snakes_.clear();
    for (const auto& [name, data] : meta.items()) {
        std::ifstream points_file(data["points"].get<std::string>());
        if (!points_file.is_open()) {
            continue;
        }

        json points_json;
        try {
            points_file >> points_json;
        }
        catch (...) {
            continue;
        }

        SnakeFeatures features = jsonToFeatures(points_json);
        features.image_paths = data["images"].get<std::vector<std::string>>();
        snakes_[name] = features;
    }

    return true;
}

bool SnakeDatabase::exportTo(const std::string& file_path) const {
    json export_data;
    for (const auto& [name, features] : snakes_) {
        export_data[name] = {
            {"keypoints", featuresToJson(features)["keypoints"]},
            {"descriptors", featuresToJson(features)["descriptors"]},
            {"images", features.image_paths}
        };
    }

    std::ofstream out_file(file_path);
    if (!out_file.is_open()) {
        return false;
    }
    out_file << export_data.dump(4);
    return true;
}

bool SnakeDatabase::importFrom(const std::string& file_path) {
    std::ifstream in_file(file_path);
    if (!in_file.is_open()) {
        return false;
    }

    json import_data;
    try {
        in_file >> import_data;
    }
    catch (...) {
        return false;
    }

    for (const auto& [name, data] : import_data.items()) {
        SnakeFeatures features;
        features.name = name;

        // Восстанавливаем ключевые точки
        for (const auto& kp_json : data["keypoints"]) {
            cv::KeyPoint kp;
            kp.pt.x = kp_json["x"].get<float>();
            kp.pt.y = kp_json["y"].get<float>();
            kp.size = kp_json["size"].get<float>();
            kp.angle = kp_json["angle"].get<float>();
            kp.response = kp_json["response"].get<float>();
            kp.octave = kp_json["octave"].get<int>();
            kp.class_id = kp_json["class_id"].get<int>();
            features.keypoints.push_back(kp);
        }

        // Восстанавливаем дескрипторы
        auto descriptors_vec = data["descriptors"].get<std::vector<uint8_t>>();
        if (!descriptors_vec.empty()) {
            features.descriptors = cv::Mat(
                descriptors_vec.size() / 128, 128, CV_32F,
                const_cast<uint8_t*>(descriptors_vec.data())).clone();
        }

        // Восстанавливаем пути к изображениям
        features.image_paths = data["images"].get<std::vector<std::string>>();

        snakes_[name] = features;
    }

    return true;
}

size_t SnakeDatabase::count() const {
    return snakes_.size();
}

std::map<std::string, size_t> SnakeDatabase::getStatistics() const {
    std::map<std::string, size_t> stats;
    for (const auto& [name, features] : snakes_) {
        stats[name] = features.keypoints.size();
    }
    return stats;
}

// Вспомогательные методы
std::string SnakeDatabase::generateImagePath(const std::string& snake_name, int index) const {
    return db_path_ + "/data/images/" + snake_name + "/photo_" + std::to_string(index) + ".jpg";
}

json SnakeDatabase::featuresToJson(const SnakeFeatures& features) const {
    json j;
    j["name"] = features.name;

    // Сохраняем ключевые точки
    for (const auto& kp : features.keypoints) {
        j["keypoints"].push_back({
            {"x", kp.pt.x},
            {"y", kp.pt.y},
            {"size", kp.size},
            {"angle", kp.angle},
            {"response", kp.response},
            {"octave", kp.octave},
            {"class_id", kp.class_id}
            });
    }

    // Сохраняем дескрипторы
    std::vector<uint8_t> descriptors_vec;
    if (!features.descriptors.empty()) {
        descriptors_vec.assign(
            features.descriptors.datastart,
            features.descriptors.dataend
        );
    }
    j["descriptors"] = descriptors_vec;

    return j;
}

SnakeFeatures SnakeDatabase::jsonToFeatures(const json& j) const
{
    SnakeFeatures features;

    try {
        // Проверяем наличие обязательных полей
        if (!j.contains("name") || !j.contains("keypoints") || !j.contains("descriptors")) {
            throw std::runtime_error("Invalid JSON structure for SnakeFeatures");
        }

        // Чтение имени
        features.name = j["name"].get<std::string>();

        // Чтение ключевых точек
        if (j["keypoints"].is_array()) {
            features.keypoints.clear();
            for (const auto& kp_json : j["keypoints"]) {
                cv::KeyPoint kp;
                kp.pt.x = kp_json["x"].get<float>();
                kp.pt.y = kp_json["y"].get<float>();
                kp.size = kp_json["size"].get<float>();
                kp.angle = kp_json["angle"].get<float>();
                kp.response = kp_json["response"].get<float>();
                kp.octave = kp_json["octave"].get<int>();
                kp.class_id = kp_json["class_id"].get<int>();
                features.keypoints.push_back(kp);
            }
        }

        // Безопасное чтение дескрипторов
        if (j["descriptors"].is_array()) {
            auto descriptors_vec = j["descriptors"].get<std::vector<uint8_t>>();

            // Проверяем размер дескрипторов (для SIFT обычно 128 элементов на ключевую точку)
            if (!descriptors_vec.empty() && !features.keypoints.empty() &&
                descriptors_vec.size() == features.keypoints.size() * 128)
            {
                features.descriptors = cv::Mat(
                    features.keypoints.size(), // строки = количество ключевых точек
                    128,                      // колонки = размер дескриптора (128 для SIFT)
                    CV_32F,                   // тип данных
                    cv::Scalar(0));           // инициализация нулями

                // Копируем данные с проверкой
                if (descriptors_vec.size() == features.descriptors.total() * features.descriptors.elemSize()) {
                    std::memcpy(features.descriptors.data,
                        descriptors_vec.data(),
                        descriptors_vec.size());
                }
                else {
                    throw std::runtime_error("Descriptor size mismatch");
                }
            }
        }
    }
    catch (const json::exception& e) {
        std::cerr << "JSON error: " << e.what() << std::endl;
        features = SnakeFeatures(); // Возвращаем пустую структуру
    }
    catch (const std::exception& e) {
        std::cerr << "Error in jsonToFeatures: " << e.what() << std::endl;
        features = SnakeFeatures();
    }

    return features;
}

bool SnakeDatabase::saveImage(const cv::Mat& image, const std::string& path) const {
    if (image.empty()) {
        return false;
    }

    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.push_back(95);

    return cv::imwrite(path, image, compression_params);
}