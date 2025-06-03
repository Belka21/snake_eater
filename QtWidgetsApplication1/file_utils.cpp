#include "file_utils.h"
#include "image_comparison.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>

using namespace std;


std::string FileUtils::getJustFileName(const std::string& fullPath) {
    fs::path p(fullPath);
    return p.filename().string();
}

std::string FileUtils::getBaseName(const std::string& filename) {
    fs::path p(filename);
    std::string stem = p.stem().string();
    size_t underscore_pos = stem.find('_');
    return (underscore_pos != std::string::npos) ? stem.substr(0, underscore_pos) : stem;
}

std::vector<std::string> FileUtils::findImageFiles(const std::string& directoryPath,
    const std::vector<std::string>& extensions) {
    std::vector<std::string> imageFiles;

    if (!directoryExists(directoryPath)) {
        std::cerr << "Directory does not exist: " << directoryPath << std::endl;
        return imageFiles;
    }

    try {
        for (const auto& entry : fs::directory_iterator(directoryPath)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
                    return std::tolower(c);
                    });

                if (extensions.empty() ||
                    std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                    imageFiles.push_back(entry.path().string());
                }
            }
        }
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "General error: " << e.what() << std::endl;
    }

    return imageFiles;
}

bool FileUtils::directoryExists(const std::string& path) {
    return fs::exists(path) && fs::is_directory(path);
}

bool FileUtils::isSameSource(const std::string& file1, const std::string& file2) {
    return getBaseName(file1) == getBaseName(file2);
}

bool FileUtils::writeCSV(const string& filename,
    const vector<string>& headers,
    const vector<vector<string>>& data) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return false;
    }

    // Запись заголовков
    if (!headers.empty()) {
        for (size_t i = 0; i < headers.size(); ++i) {
            file << headers[i];
            if (i != headers.size() - 1) file << ",";
        }
        file << "\n";
    }

    // Запись данных
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i != row.size() - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
    return true;
}

bool FileUtils::appendCSV(const string& filename,
    const vector<string>& row) {
    ofstream file(filename, ios::app);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return false;
    }

    for (size_t i = 0; i < row.size(); ++i) {
        file << row[i];
        if (i != row.size() - 1) file << ",";
    }
    file << "\n";

    file.close();
    return true;
}

vector<vector<string>> FileUtils::readCSV(const string& filename,
    char delimiter) {
    vector<vector<string>> result;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return result;
    }

    string line;
    while (getline(file, line)) {
        vector<string> row;
        stringstream ss(line);
        string cell;

        while (getline(ss, cell, delimiter)) {
            row.push_back(cell);
        }

        if (!row.empty()) {
            result.push_back(row);
        }
    }

    file.close();
    return result;
}

bool FileUtils::createCSV(const string& filename,
    const vector<string>& headers) {
    ofstream file(filename);
    if (!file.is_open()) return false;

    if (!headers.empty()) {
        for (size_t i = 0; i < headers.size(); ++i) {
            file << headers[i];
            if (i != headers.size() - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
    return true;
}

bool FileUtils::saveComparisonResults(const string& filename,
    const vector<struct ComparisonResult>& results) {
    
    //ofstream csv_file(filename);
    //// CSV header
    //csv_file << "ImageA,ImageB,IsSameSource,SIFT_KP_A,SIFT_KP_B,ORB_KP_A,ORB_KP_B,"
    //    << "SIFT_Matches,SIFT_Good_Matches,SIFT_Avg_Distance,SIFT_Result,"
    //    << "ORB_Matches,ORB_Good_Matches,ORB_Avg_Distance,ORB_Result,"
    //    << "SIFT_Detect_Time,ORB_Detect_Time,SIFT_Match_Time,ORB_Match_Time\n";

    //// Data
    //for (const auto& result : results) {
    //    csv_file << result.imageA << ","
    //        << result.imageB << ","
    //        << (result.is_same_source ? "YES" : "NO") << ","
    //        << result.sift_kpA << ","
    //        << result.sift_kpB << ","
    //        << result.orb_kpA << ","
    //        << result.orb_kpB << ","
    //        << result.sift_matches << ","
    //        << result.sift_good_matches << ","
    //        << result.sift_avg_distance << ","
    //        << (result.sift_result ? "YES" : "NO") << ","
    //        << result.orb_matches << ","
    //        << result.orb_good_matches << ","
    //        << result.orb_avg_distance << ","
    //        << (result.orb_result ? "YES" : "NO") << ","
    //        << result.sift_detect_time << ","
    //        << result.orb_detect_time << ","
    //        << result.sift_match_time << ","
    //        << result.orb_match_time << "\n";
    //}

    //csv_file.close();

    return true;
}