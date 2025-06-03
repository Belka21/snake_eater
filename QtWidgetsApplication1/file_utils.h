#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

class FileUtils {
public:
    /* Функции для работы с файлами */
    static std::string getBaseName(const std::string& filename);
    static std::string getJustFileName(const std::string& fullPath);
    static std::vector<std::string> findImageFiles(const std::string& directoryPath,
        const std::vector<std::string>& extensions = { ".jpg", ".jpeg", ".png" });
    static bool directoryExists(const std::string& path);
    static bool isSameSource(const std::string& file1, const std::string& file2);

    /* Функции для работы с CSV */
    static bool writeCSV(const std::string& filename,
        const std::vector<std::string>& headers,
        const std::vector<std::vector<std::string>>& data);

    static bool appendCSV(const std::string& filename,
        const std::vector<std::string>& row);

    static std::vector<std::vector<std::string>> readCSV(const std::string& filename,
        char delimiter = ',');

    static bool createCSV(const std::string& filename,
        const std::vector<std::string>& headers = {});

    /* Специализированные функции для сравнения изображений */
    static bool saveComparisonResults(const std::string& filename,
        const std::vector<struct ComparisonResult>& results);
};

#endif // FILE_UTILS_H