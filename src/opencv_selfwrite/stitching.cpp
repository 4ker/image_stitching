#include <unistd.h>     // fork
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>    // max, min, sort
#include <set>
#include <random>
#include <queue>        // priority_queue
#include <cmath>        // sqrt
#include <utility>      // pair
#include <limits>       // max

#include "opencv2/opencv.hpp"       // warpPerspective
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#define MIN_MATCHES 5
#define MIN_GOOD_MATCHES 25

#define FORK_OUTPUT true


using namespace cv;


class imageData {
public:
    Mat color;
    Mat grey;
    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    std::vector<DMatch> matches;
    Mat homography;
    Mat homography_inv;
};

void help();

void loadImage(const char *filename, Mat& colorMat, Mat& greyMat) {
    colorMat = imread(filename);
    cv::cvtColor(colorMat, greyMat, CV_BGR2GRAY);
}

auto featureDetect(const Mat& image) {
    static const cv::SiftFeatureDetector detector (0.05, 5.0);
    std::vector<KeyPoint> keypoints;
    detector.detect(image, keypoints);
    return keypoints;
}

auto featureExtract(const Mat& image, std::vector<KeyPoint>& keypoints) {
    static const cv::SiftDescriptorExtractor extractor (3.0);
    Mat descriptors;
    extractor.compute(image, keypoints, descriptors);
    return descriptors;
}

auto KNNMatch(const Mat& descriptors1, const Mat& descriptors2, const unsigned int knn = 1) {

    ////////////////////
    // K-NN
    ////////////////////
    //
    // DMatch
    //     http://docs.opencv.org/modules/features2d/doc/common_interfaces_of_descriptor_matchers.html
    //     .queryIdx : descriptors1's index
    //     .trainIdx : descriptors2's index
    //     .distance
    //
    // descriptors : [ 128 x N ]    (SIFT)

    std::vector<DMatch> matches;

    for (unsigned int i = 0; i < (unsigned int) descriptors1.rows; i++) {

        std::priority_queue<std::pair<double, DMatch> > min_heap;

        for (unsigned int j = 0; j < (unsigned int) descriptors2.rows; j++) {
            // calculate distance

            const auto distance = norm(descriptors1.row(i), descriptors2.row(j), NORM_L2);

            DMatch match;
            match.queryIdx = i;
            match.trainIdx = j;
            match.distance = distance;

            min_heap.push(std::make_pair(distance, match));

            if (min_heap.size() > knn) {
                min_heap.pop();
            }
        }

        for (unsigned int j = 0; j < knn; j++) {
            auto& value = min_heap.top();
            matches.push_back(value.second);
            min_heap.pop();
        }
    }

    return matches;
}

bool findMatches(imageData& image_obj, imageData& image_scene) {

    auto& matches = image_obj.matches;
    matches = KNNMatch(image_obj.descriptors, image_scene.descriptors);

    {
        double max_dist = 0, min_dist = std::numeric_limits<double>::max();

        // calculation of max and min distances between keypoints
        double dist;

        // descriptors : [ 128 x N ]

        for (unsigned int k = 0; k < matches.size(); k++) {
            dist = matches[k].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        // Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
        std::vector<DMatch> good_matches;

        // good matches threshold
        if (min_dist < MIN_GOOD_MATCHES) {

            for (unsigned int k = 0; k < matches.size(); k++) {
                if (matches[k].distance < 2*min_dist) {
                    good_matches.push_back(matches[k]);
                }
            }

            matches = std::move(good_matches);

            return true;
        }

        matches = std::move(good_matches);

        return false;
    }
}

auto matchPoint(const Mat& matrix, const int x, const int y) {
    Mat source (3, 1, matrix.type(), Scalar(1));
    source.at<double>(0, 0) = x;
    source.at<double>(0, 1) = y;
    Mat result = matrix * source;

    result = result / result.at<double>(0, 2);
    Point2f point (result.at<double>(0, 0), result.at<double>(0, 1));
    return point;
}

auto calcHomography(std::vector<Point2f>& data, std::vector<Point2f>& target) {
    return getPerspectiveTransform(data, target);
}

Mat ransac(std::vector<Point2f>& points,
           std::vector<Point2f>& points_target,
           const unsigned int threshold = 3,
           const unsigned int iteration = 1000) {

    unsigned int best_total_inliner = 0;
    double best_distance = std::numeric_limits<double>::max();
    Mat best_model;

    for (unsigned int i = 0; i < iteration; i++) {

        // we need 4 pair of points for homographies
        // (A -> A'), (B -> B'), (C -> C'), (D -> D')

        std::vector<Point2f> sample_source;
        std::vector<Point2f> sample_target;
        std::set<unsigned int> sample_set;

        std::random_device rd;
        std::default_random_engine gen(rd());
        std::uniform_int_distribution<unsigned int> dis(1, points.size());
        while (sample_set.size() < 4) {
            const auto value = dis(gen);
            if (sample_set.count(value-1) == 0) {
                sample_set.emplace(value-1);
                sample_source.push_back(points[value-1]);
                sample_target.push_back(points_target[value-1]);
            }
        }

        // calculate homography
        decltype(best_model) model = calcHomography(sample_source, sample_target);

        unsigned int total_inliner = 0;
        double total_distance = 0;

        for (unsigned int j = 0; j < points.size(); j++) {
            // source point * H = result point
            // | result point & target point | = distance
            // if distance < epislon => inliner

            const auto match_point = matchPoint(model, points[j].x, points[j].y);
            const auto diff_point = points_target[j] - match_point;
            const auto distance = diff_point.x * diff_point.x + diff_point.y * diff_point.y;

            total_distance += distance;

            if (distance < threshold) {
                total_inliner++;
            }
        }

        total_distance = sqrt(total_distance);

        if (best_model.empty() || (total_inliner > best_total_inliner) || (best_distance > total_distance)) {
            best_total_inliner = total_inliner;
            best_model = model;
            best_distance = total_distance;
        }
    }

    return best_model;
}

auto findHomoWithRANSAC(std::vector<Point2f>& points1, std::vector<Point2f>& points2) {
    // find homography with RANSAC
    return ransac(points1, points2);
}

void outputImage(const Mat& image, const std::string filename) {
    static const vector<int> compression_params (1, CV_IMWRITE_JPEG_QUALITY);
    imwrite(filename + ".jpg", image, compression_params);
}

auto stitching(const std::vector<unsigned int>& group,
               const std::vector<imageData>& images) {

    const auto middle = group.size() / 2;
    const auto& image_middle = images[group[middle]].color;

    // count edges

    auto height_minmax = std::minmax({ (const float) 0 });
    auto width_minmax = std::minmax({ (const float) 0 });

    for (unsigned int i = 0; i < group.size(); i++) {
        if (i == middle)
            continue;

        const auto& image1 = images[group[i]].grey;
        const auto& image2 = images[group[middle]].grey;
        const auto& homography = images[group[i]].homography;

        // version 1
        std::vector<Point2f> obj_corners(4);
        obj_corners[0] = cvPoint(0, 0);
        obj_corners[1] = cvPoint(image1.cols, 0);
        obj_corners[2] = cvPoint(image1.cols, image1.rows);
        obj_corners[3] = cvPoint(0, image1.rows);
        std::vector<Point2f> scene_corners(4);

        perspectiveTransform(obj_corners, scene_corners, homography);

        // version 2
        // const auto point1 = matchPoint(homography, 0, 0);
        // const auto point2 = matchPoint(homography, image1.cols, 0);
        // const auto point3 = matchPoint(homography, image1.cols, image1.rows);
        // const auto point4 = matchPoint(homography, 0, image1.rows);

        height_minmax = std::minmax({ height_minmax.first,
                                      height_minmax.second,
                                      (const float) image2.rows,
                                      scene_corners[0].y,
                                      scene_corners[1].y,
                                      scene_corners[2].y,
                                      scene_corners[3].y });

        width_minmax = std::minmax({ width_minmax.first,
                                     width_minmax.second,
                                     (const float) image2.cols,
                                     scene_corners[0].x,
                                     scene_corners[1].x,
                                     scene_corners[2].x,
                                     scene_corners[3].x });
    }

    const int up = height_minmax.first;
    const int down = height_minmax.second;
    const int left = width_minmax.first;
    const int right = width_minmax.second;

    const auto height = down - up;
    const auto width = right - left;

    Mat result (height,    // rows
                width,     // cols
                image_middle.type(),    // type
                Scalar(0, 0, 0));       // default

    image_middle.copyTo(result(cv::Rect(-left,  // x
                                        -up,    // y
                                        image_middle.cols,     // width
                                        image_middle.rows)));  // height

    for (int y = 0; y < result.rows; y++) {
        for (int x = 0; x < result.cols; x++) {
            for (auto& index : group) {
                if (index == group[middle])
                    continue;

                const auto& image1 = images[index].color;

                const auto point = matchPoint(images[index].homography_inv, x + left, y + up);
                if ((image1.cols > point.x) && (point.x > 0) && (image1.rows > point.y) && (point.y > 0)) {

                    // use Vec3b to keep the channels with BGR (not Vec3d ...)
                    // http://docs.opencv.org/doc/user_guide/ug_mat.html#basic-operations-with-images
                    // consider a 3 channel image with BGR color ordering (default of imread)
                    result.at<cv::Vec3b>(y, x) = image1.at<cv::Vec3b>(point.y, point.x);
                }
            }
        }
    }

    return result;
}

int main(int argc, char* argv[]) {

    if (argc < 3) {
        help();
        return -1;
    }

    // initial

    const unsigned int totals = argc-1;
    auto images_name = &(argv[1]);

    std::vector<imageData> images (totals);

    // before groups

    for (unsigned int i = 0; i < totals; i++) {
        Mat colorMat;
        Mat greyMat;
        loadImage(images_name[i], colorMat, greyMat);
        images[i].color = std::move(colorMat);
        images[i].grey  = std::move(greyMat);
        images[i].keypoints = featureDetect(images[i].grey);
        images[i].descriptors = featureExtract(images[i].grey, images[i].keypoints);
    }


    // grouping

    std::vector<std::vector<unsigned int> > groups;

    std::set<unsigned int> images_set;
    for (unsigned int i = 0; i < totals; i++) {
        images_set.emplace(i);
    }

    std::map<unsigned int, unsigned int> get_group;
    for (unsigned int i = 0; i < images.size(); i++) {
        for (unsigned int j = 0; j < images.size(); j++) {
            // escape same image
            if (i == j)
                continue;

            // escape already found groups
            if ((get_group[i] > 0) && (get_group[j] > 0))
                continue;

            bool result = findMatches(images[j], images[i]);

            if (result && (images[j].matches.size() > MIN_MATCHES)) {
                unsigned int index = 0;
                unsigned int value = 0;

                if (get_group[i] > 0) {
                    index = get_group[i];
                    value = j;
                } else if (get_group[j] > 0) {
                    index = get_group[j];
                    value = i;
                }

                if (index > 0) {
                    groups[index-1].push_back(value);
                    get_group[i] = index;
                    get_group[j] = index;
                } else {
                    index = groups.size()+1;
                    get_group[i] = index;
                    get_group[j] = index;

                    std::vector<unsigned int> group (2);
                    group[0] = i;
                    group[1] = j;
                    groups.push_back(std::move(group));
                }
            }
        }
    }

    for (unsigned int i = 0; i < groups.size(); i++) {

#ifdef FORK_OUTPUT
        auto pid = fork();

        if (pid > 0) {
            // parent
            continue;
        }
#endif

        auto& group = groups[i];

        // find order
        //sort_group(group, images);

        const auto middle = group.size() / 2;

        // two images per calculation
        for (unsigned int j = 0; j < group.size(); j++) {
            if (j == middle)
                continue;

            const auto index = group[j];
            const auto middle_index = group[middle];

            auto& image_obj = images[index];
            auto& image_scene = images[middle_index];
            auto& matches = image_obj.matches;

            findMatches(image_obj, image_scene);

            // Localize the object

            std::vector<Point2f> obj;
            std::vector<Point2f> scene;

            for (unsigned int k = 0; k < matches.size(); k++) {
                // Get the keypoints from the good matches
                obj.push_back(image_obj.keypoints[ matches[k].queryIdx ].pt);
                scene.push_back(image_scene.keypoints[ matches[k].trainIdx ].pt);
            }


            // Calculate Homography with RANSAC


            // find Homography

            image_obj.homography = findHomoWithRANSAC(obj, scene);
            //image_obj.homography = findHomography(obj, scene, CV_RANSAC);

            image_obj.homography_inv = image_obj.homography.inv();

        }

        // stitching
        const auto result_image = stitching(group, images);

        std::string filename = "result-";
        filename += std::to_string(i);
        outputImage(result_image, filename);


#ifdef FORK_OUTPUT
        break;
#endif

    }

    return 0;
}

void help() {
    printf(" Usage: <img1> <img2> [<img3> ...]\n");
}
