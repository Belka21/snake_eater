#ifndef IMAGE_COMPARISON_H
#define IMAGE_COMPARISON_H

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include <string>
#include <chrono>
#include "file_utils.h"

struct FeaturePoints {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    double processing_time;
};

struct MatchResult {
    int total_matches;
    int good_matches;
    float avg_distance;
    bool is_match;
    double matching_time;
};

struct ComparisonResult {
    std::string imageA;
    std::string imageB;
    bool is_same_source;

    FeaturePoints siftA, siftB;
    FeaturePoints orbA, orbB;

    MatchResult sift_match;
    MatchResult orb_match;
};

// Feature Detection
FeaturePoints detectSIFTFeatures(const cv::Mat& image);
FeaturePoints detectORBFeatures(const cv::Mat& image);

// Feature Matching
MatchResult matchFeatures(const FeaturePoints& featuresA,
    const FeaturePoints& featuresB,
    int normType,
    float ratio_threshold);

// Main comparison function
void compareImages(const cv::Mat& imgA, const cv::Mat& imgB,
    const std::string& imgA_path, const std::string& imgB_path);

// Helper functions
void initResult(ComparisonResult& result,
    const std::string& imgA_path, const std::string& imgB_path);
std::vector<cv::DMatch> filterMatchesWithHomography(
    const std::vector<cv::KeyPoint>& kp1,
    const std::vector<cv::KeyPoint>& kp2,
    const std::vector<cv::DMatch>& matches,
    double ransacThreshold = 3.0);

// Results handling
void saveResultsToCSV(const std::string& filename);

#endif // IMAGE_COMPARISON_H