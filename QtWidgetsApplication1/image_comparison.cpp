#include "image_comparison.h"
#include <chrono>

using namespace cv;
using namespace std;
using namespace chrono;

vector<ComparisonResult> results;

FeaturePoints detectSIFTFeatures(const Mat& image) {
    FeaturePoints result;
    Ptr<SIFT> detector = SIFT::create();

    auto start = high_resolution_clock::now();
    detector->detectAndCompute(image, noArray(),
        result.keypoints, result.descriptors);
    auto stop = high_resolution_clock::now();

    result.processing_time = duration_cast<milliseconds>(stop - start).count() / 1000.0;
    return result;
}

FeaturePoints detectORBFeatures(const Mat& image) {
    FeaturePoints result;
    Ptr<ORB> detector = ORB::create(1000);

    auto start = high_resolution_clock::now();
    detector->detectAndCompute(image, noArray(),
        result.keypoints, result.descriptors);
    auto stop = high_resolution_clock::now();

    result.processing_time = duration_cast<milliseconds>(stop - start).count() / 1000.0;
    return result;
}

MatchResult matchFeatures(const FeaturePoints& featuresA,
    const FeaturePoints& featuresB,
    int normType,
    float ratio_threshold)
{
    MatchResult result;
    result.total_matches = 0;
    result.good_matches = 0;
    result.avg_distance = 0;
    result.is_match = false;

    try {
        if (featuresA.descriptors.empty() || featuresB.descriptors.empty()) {
            return result;
        }

        // Используем FlannBasedMatcher для SIFT/SURF
        cv::Ptr<cv::DescriptorMatcher> matcher;
        if (normType == cv::NORM_L2) {
            matcher = cv::FlannBasedMatcher::create();
        }
        else {
            matcher = cv::BFMatcher::create(normType);
        }

        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(
            featuresA.descriptors,
            featuresB.descriptors,
            knn_matches,
            2
        );

        // Фильтр по соотношению расстояний
        std::vector<cv::DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i].size() < 2) continue;

            if (knn_matches[i][0].distance <
                ratio_threshold * knn_matches[i][1].distance) {
                good_matches.push_back(knn_matches[i][0]);
            }
        }

        // Фильтр по гомографии
        std::vector<cv::DMatch> inliers = filterMatchesWithHomography(
            featuresA.keypoints,
            featuresB.keypoints,
            good_matches
        );

        // Заполняем результат
        result.total_matches = good_matches.size();
        result.good_matches = inliers.size();

        if (!inliers.empty()) {
            float avg_dist = 0;
            for (const auto& m : inliers) {
                avg_dist += m.distance;
            }
            result.avg_distance = avg_dist / inliers.size();
        }

        result.is_match = !inliers.empty();
    }
    catch (...) {
        // В случае ошибки возвращаем пустой результат
    }

    return result;
}

vector<DMatch> filterMatchesWithHomography(
    const vector<KeyPoint>& kp1,
    const vector<KeyPoint>& kp2,
    const vector<DMatch>& matches,
    double ransacThreshold) {

    if (matches.size() < 4) return matches;

    vector<Point2f> pts1, pts2;
    for (const auto& m : matches) {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }

    Mat mask;
    findHomography(pts1, pts2, RANSAC, ransacThreshold, mask);

    vector<DMatch> inliers;
    for (size_t i = 0; i < matches.size(); i++) {
        if (mask.empty() || mask.at<uchar>(i)) {
            inliers.push_back(matches[i]);
        }
    }

    return inliers;
}

void compareImages(const Mat& imgA, const Mat& imgB,
    const string& imgA_path, const string& imgB_path) {
    ComparisonResult result;
    initResult(result, imgA_path, imgB_path);

    // Detect features
    result.siftA = detectSIFTFeatures(imgA);
    result.siftB = detectSIFTFeatures(imgB);
    result.orbA = detectORBFeatures(imgA);
    result.orbB = detectORBFeatures(imgB);

    // Match features
    result.sift_match = matchFeatures(result.siftA, result.siftB, NORM_L2, 0.7);
    result.orb_match = matchFeatures(result.orbA, result.orbB, NORM_HAMMING, 0.6);

    results.push_back(result);
}

void initResult(ComparisonResult& result,
    const string& imgA_path, const string& imgB_path) {
    result.imageA = FileUtils::getJustFileName(imgA_path);
    result.imageB = FileUtils::getJustFileName(imgB_path);
    result.is_same_source = FileUtils::isSameSource(imgA_path, imgB_path);
}

void saveResultsToCSV(const string& filename) {
    vector<vector<string>> data;
    vector<string> headers = {
        "ImageA", "ImageB", "IsSameSource",
        "SIFT_KP_A", "SIFT_KP_B", "ORB_KP_A", "ORB_KP_B",
        "SIFT_Matches", "SIFT_Good_Matches", "SIFT_Avg_Distance", "SIFT_Result",
        "ORB_Matches", "ORB_Good_Matches", "ORB_Avg_Distance", "ORB_Result",
        "SIFT_Detect_Time", "ORB_Detect_Time", "SIFT_Match_Time", "ORB_Match_Time"
    };

    for (const auto& result : results) {
        data.push_back({
            result.imageA,
            result.imageB,
            result.is_same_source ? "YES" : "NO",
            to_string(result.siftA.keypoints.size()),
            to_string(result.siftB.keypoints.size()),
            to_string(result.orbA.keypoints.size()),
            to_string(result.orbB.keypoints.size()),
            to_string(result.sift_match.total_matches),
            to_string(result.sift_match.good_matches),
            to_string(result.sift_match.avg_distance),
            result.sift_match.is_match ? "YES" : "NO",
            to_string(result.orb_match.total_matches),
            to_string(result.orb_match.good_matches),
            to_string(result.orb_match.avg_distance),
            result.orb_match.is_match ? "YES" : "NO",
            to_string(result.siftA.processing_time + result.siftB.processing_time),
            to_string(result.orbA.processing_time + result.orbB.processing_time),
            to_string(result.sift_match.matching_time),
            to_string(result.orb_match.matching_time)
            });
    }

    FileUtils::writeCSV(filename, headers, data);
    cout << "Results saved to " << filename << endl;
}