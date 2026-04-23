//
// Created by gaoxiang on 19-5-2.
//

#include <opencv2/opencv.hpp>

#include "myslam/algorithm.h"
#include "myslam/backend.h"
#include "myslam/config.h"
#include "myslam/feature.h"
#include "myslam/frontend.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/viewer.h"

namespace myslam {

Frontend::Frontend() {
    // Shi-Tomasi corner detector with parameters from config
    gftt_ =
        cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);
    num_features_init_ = Config::Get<int>("num_features_init");
    num_features_ = Config::Get<int>("num_features");
}

/**
 * Add a new frame to the frontend. Depending on the current status, 
 * it will either initialize the map, track the current frame, or reset 
 * if tracking is lost.
 * @param frame The new frame to be added and processed by the frontend.
 * @return true if the frame was successfully processed, false otherwise.
 */
bool Frontend::AddFrame(myslam::Frame::Ptr frame) {
    this->current_frame_ = frame;

    switch (this->status_) {
        case FrontendStatus::INITING:
            StereoInit(); // Attempt stereo initialization on the first frame
            break;
        case FrontendStatus::TRACKING_GOOD:
        case FrontendStatus::TRACKING_BAD:
            Track(); // Track the current frame against the last frame and update status
            break;
        case FrontendStatus::LOST:
            Reset(); // Reset the frontend if tracking is lost
            break;
    }

    this->last_frame_ = current_frame_;
    return true;
}

bool Frontend::Track() {
    // 如果存在上一帧，则用上一帧的位姿和相对运动初始化当前帧的位姿
    if (last_frame_) {
        current_frame_->SetPose(relative_motion_ * last_frame_->Pose()); // 用相对运动初始化当前帧位姿
    }

    // 1. 跟踪上一帧的特征点，得到当前帧的初始特征点集合
    int num_track_last = TrackLastFrame();
    // 2. 优化当前帧的位姿，并统计内点数量
    tracking_inliers_ = EstimateCurrentPose();

    // 3. 根据内点数量判断当前跟踪状态
    if (tracking_inliers_ > num_features_tracking_) {
        // 跟踪效果好
        status_ = FrontendStatus::TRACKING_GOOD;
    } else if (tracking_inliers_ > num_features_tracking_bad_) {
        // 跟踪效果较差
        status_ = FrontendStatus::TRACKING_BAD;
    } else {
        // 跟踪失败，进入丢失状态
        status_ = FrontendStatus::LOST;
    }

    // 4. 判断是否需要插入关键帧
    InsertKeyframe();
    // 5. 更新相对运动，用于下一帧的位姿初始化
    relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();

    // 6. 如果有可视化模块，则显示当前帧
    if (viewer_) viewer_->AddCurrentFrame(current_frame_);
    return true;
}

bool Frontend::InsertKeyframe() {
    // 1. 如果当前帧跟踪到的内点数量足够多，则不需要插入关键帧，直接返回
    if (tracking_inliers_ >= num_features_needed_for_keyframe_) {
        // 还有足够的特征点，不插入关键帧
        return false;
    }
    // 2. 当前帧被设为新的关键帧
    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_); // 加入地图

    LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
              << current_frame_->keyframe_id_;

    // 3. 为新关键帧中的地图点添加观测关系
    SetObservationsForKeyFrame();
    // 4. 在当前帧左图检测新的特征点
    DetectFeatures();

    // 5. 在右图跟踪新检测到的特征点，获得双目匹配
    FindFeaturesInRight();
    // 6. 对新检测到的特征点进行三角化，生成新的地图点
    TriangulateNewPoints();
    // 7. 通知后端进行地图优化
    backend_->UpdateMap();

    // 8. 如果有可视化模块，则更新地图显示
    if (viewer_) viewer_->UpdateMap();

    return true;
}

void Frontend::SetObservationsForKeyFrame() {
    for (auto &feat : current_frame_->features_left_) {
        auto mp = feat->map_point_.lock();
        if (mp) mp->AddObservation(feat);
    }
}

int Frontend::TriangulateNewPoints() {
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    SE3 current_pose_Twc = current_frame_->Pose().inverse();
    int cnt_triangulated_pts = 0;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_left_[i]->map_point_.expired() &&
            current_frame_->features_right_[i] != nullptr) {
            // 左图的特征点未关联地图点且存在右图匹配点，尝试三角化
            std::vector<Vec3> points{
                camera_left_->pixel2camera(
                    Vec2(current_frame_->features_left_[i]->position_.pt.x,
                         current_frame_->features_left_[i]->position_.pt.y)),
                camera_right_->pixel2camera(
                    Vec2(current_frame_->features_right_[i]->position_.pt.x,
                         current_frame_->features_right_[i]->position_.pt.y))};
            Vec3 pworld = Vec3::Zero();

            if (triangulation(poses, points, pworld) && pworld[2] > 0) {
                auto new_map_point = MapPoint::CreateNewMappoint();
                pworld = current_pose_Twc * pworld;
                new_map_point->SetPos(pworld);
                new_map_point->AddObservation(
                    current_frame_->features_left_[i]);
                new_map_point->AddObservation(
                    current_frame_->features_right_[i]);

                current_frame_->features_left_[i]->map_point_ = new_map_point;
                current_frame_->features_right_[i]->map_point_ = new_map_point;
                map_->InsertMapPoint(new_map_point);
                cnt_triangulated_pts++;
            }
        }
    }
    LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
    return cnt_triangulated_pts;
}

int Frontend::EstimateCurrentPose() {
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
        LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(
            std::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // vertex
    VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(current_frame_->Pose());
    optimizer.addVertex(vertex_pose);

    // K
    Mat33 K = camera_left_->K();

    // edges
    int index = 1;
    std::vector<EdgeProjectionPoseOnly *> edges;
    std::vector<Feature::Ptr> features;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        auto mp = current_frame_->features_left_[i]->map_point_.lock();
        if (mp) {
            features.push_back(current_frame_->features_left_[i]);
            EdgeProjectionPoseOnly *edge =
                new EdgeProjectionPoseOnly(mp->pos_, K);
            edge->setId(index);
            edge->setVertex(0, vertex_pose);
            edge->setMeasurement(
                toVec2(current_frame_->features_left_[i]->position_.pt));
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber);
            edges.push_back(edge);
            optimizer.addEdge(edge);
            index++;
        }
    }

    // estimate the Pose the determine the outliers
    const double chi2_th = 5.991;
    int cnt_outlier = 0;
    for (int iteration = 0; iteration < 4; ++iteration) {
        vertex_pose->setEstimate(current_frame_->Pose());
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cnt_outlier = 0;

        // count the outliers
        for (size_t i = 0; i < edges.size(); ++i) {
            auto e = edges[i];
            if (features[i]->is_outlier_) {
                e->computeError();
            }
            if (e->chi2() > chi2_th) {
                features[i]->is_outlier_ = true;
                e->setLevel(1);
                cnt_outlier++;
            } else {
                features[i]->is_outlier_ = false;
                e->setLevel(0);
            };

            if (iteration == 2) {
                e->setRobustKernel(nullptr);
            }
        }
    }

    LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
              << features.size() - cnt_outlier;
    // Set pose and outlier
    current_frame_->SetPose(vertex_pose->estimate());

    LOG(INFO) << "Current Pose = \n" << current_frame_->Pose().matrix();

    for (auto &feat : features) {
        if (feat->is_outlier_) {
            feat->map_point_.reset();
            feat->is_outlier_ = false;  // maybe we can still use it in future
        }
    }
    return features.size() - cnt_outlier;
}

int Frontend::TrackLastFrame() {
    // use LK flow to estimate points in the right image
    std::vector<cv::Point2f> kps_last, kps_current;
    for (auto &kp : last_frame_->features_left_) {
        if (kp->map_point_.lock()) {
            // use project point
            auto mp = kp->map_point_.lock();
            auto px =
                camera_left_->world2pixel(mp->pos_, current_frame_->Pose());
            kps_last.push_back(kp->position_.pt);
            kps_current.push_back(cv::Point2f(px[0], px[1]));
        } else {
            kps_last.push_back(kp->position_.pt);
            kps_current.push_back(kp->position_.pt);
        }
    }

    std::vector<uchar> status;
    Mat error;
    cv::calcOpticalFlowPyrLK(
        last_frame_->left_img_, current_frame_->left_img_, kps_last,
        kps_current, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    int num_good_pts = 0;

    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            cv::KeyPoint kp(kps_current[i], 7);
            Feature::Ptr feature(new Feature(current_frame_, kp));
            feature->map_point_ = last_frame_->features_left_[i]->map_point_;
            current_frame_->features_left_.push_back(feature);
            num_good_pts++;
        }
    }

    LOG(INFO) << "Find " << num_good_pts << " in the last image.";
    return num_good_pts;
}

// Stereo camera initialization: detect features, match stereo, build initial map
bool Frontend::StereoInit() {
    // 1. 在左图检测特征点
    int num_features_left = DetectFeatures();
    // 2. 在右图用光流跟踪左图特征点，获得双目匹配
    int num_coor_features = FindFeaturesInRight();
    // 3. 如果匹配特征点数不足，初始化失败
    // 这里的num_features_init_是一个阈值，表示至少需要多少个匹配特征点才能进行可靠的初始化
    if (num_coor_features < num_features_init_) { 
        return false;
    }

    // 4. 用双目匹配点三角化，建立初始地图
    bool build_map_success = BuildInitMap();
    if (build_map_success) {
        // 5. 初始化成功，切换到跟踪状态
        status_ = FrontendStatus::TRACKING_GOOD;
        // 6. 可视化：显示当前帧和地图
        if (viewer_) {
            viewer_->AddCurrentFrame(current_frame_);
            viewer_->UpdateMap();
        }
        return true;
    }
    // 7. 初始化失败
    return false;
}

int Frontend::DetectFeatures() {
    // 创建一个与当前帧左图同样大小的掩码（mask），初始值全为255（有效区域），用于后续特征检测，掩码为0的区域不会被检测为新特征。
    cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);
    
    // 遍历当前帧左图中已存在的特征点，以每个特征点为中心，画一个 20×20 的矩形区域（左上角−10，右下角+10），并将该区域的掩码值设为0（无效）
    // 防止新检测的特征点与已有特征点距离过近，保证空间分布均匀。
    for (auto &feat : current_frame_->features_left_) {
        cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                      feat->position_.pt + cv::Point2f(10, 10), 0, cv::FILLED);
    }

    // 使用 Shi-Tomasi 角点检测器（GFTTDetector）在当前帧左图中检测特征点，传入之前创建的掩码（mask）以确保新特征点不会出现在已有特征点附近。检测到的特征点存储在 keypoints 向量中。
    std::vector<cv::KeyPoint> keypoints;
    gftt_->detect(current_frame_->left_img_, keypoints, mask);
    
    // 将检测到的特征点转换为 Feature 对象，并关联到当前帧的 features_left_ 向量中。同时统计检测到的新特征点数量并返回。
    int cnt_detected = 0;
    for (auto &kp : keypoints) {
        current_frame_->features_left_.push_back(
            Feature::Ptr(new Feature(current_frame_, kp)));
        cnt_detected++;
    }

    LOG(INFO) << "Detect " << cnt_detected << " new features";
    return cnt_detected;
}

int Frontend::FindFeaturesInRight() {
    // 1. 构建左图和右图的特征点初始位置列表
    std::vector<cv::Point2f> kps_left, kps_right;
    for (auto &kp : current_frame_->features_left_) {
        kps_left.push_back(kp->position_.pt); // 左图特征点像素坐标
        // lock() 是用于 std::weak_ptr 的成员函数。它的作用是尝试将 weak_ptr 升级为 std::shared_ptr。
        // 如果该特征点已经关联了一个地图点（map point），则尝试获取该地图点的共享指针。
        // 如果成功获取到地图点，则使用该地图点的三维位置通过相机模型投影到右图像平面上，得到右图特征点的初始像素坐标。
        // 否则，如果该特征点没有关联地图点，则直接使用左图特征点的像素坐标作为右图特征点的初始猜测。
        auto mp = kp->map_point_.lock(); // 世界坐标系中的地图点
        if (mp) {
            // 如果该特征点已关联地图点，则用三维点投影到右图作为初始猜测
            auto px = camera_right_->world2pixel(mp->pos_, current_frame_->Pose());
            kps_right.push_back(cv::Point2f(px[0], px[1]));
        } else {
            // 使用左图的像素点坐标作为右图的初始猜测坐标
            kps_right.push_back(kp->position_.pt);
        }
    }

    // 2. 使用金字塔LK光流法在右图中跟踪左图特征点
    std::vector<uchar> status; // 跟踪状态（1=成功，0=失败）
    Mat error; // 跟踪误差
    cv::calcOpticalFlowPyrLK(
        current_frame_->left_img_,   // 输入：左图
        current_frame_->right_img_,  // 输入：右图
        kps_left,                    // 输入：左图特征点
        kps_right,                   // 输入输出：右图特征点（初始猜测/输出结果）
        status,                      // 输出：每个点的跟踪状态
        error,                       // 输出：每个点的误差
        cv::Size(11, 11),            // 光流窗口大小
        3,                           // 金字塔层数
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), // 终止条件
        cv::OPTFLOW_USE_INITIAL_FLOW // 使用初始猜测
    );

    int num_good_pts = 0;
    // 3. 处理跟踪结果，将成功跟踪的点封装为右图特征对象
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            // 跟踪成功，创建右图特征对象并加入当前帧
            cv::KeyPoint kp(kps_right[i], 7); // 7为特征点半径（特征点不是一个像素，而是一块区域）
            Feature::Ptr feat(new Feature(current_frame_, kp)); // 创建右图特征对象
            feat->is_on_left_image_ = false; // 标记这个特征点来自右图，下面已经加了，这里加入可能是为了健壮性
            current_frame_->features_right_.push_back(feat);
            num_good_pts++;
        } else {
            // 跟踪失败，右图特征为nullptr
            current_frame_->features_right_.push_back(nullptr);
        }
    }
    LOG(INFO) << "Find " << num_good_pts << " in the right image.";
    return num_good_pts;
}

bool Frontend::BuildInitMap() {
    // 1. 获取左右目相机的位姿，用于三角化
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    size_t cnt_init_landmarks = 0; // 记录新建地图点数量

    // 2. 遍历当前帧左图的所有特征点
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        // 如果右图没有对应的特征点，则跳过
        if (current_frame_->features_right_[i] == nullptr) continue;

        // 3. 将左右图的像素坐标转换为相机坐标系下的归一化坐标
        std::vector<Vec3> points{
            camera_left_->pixel2camera(
                Vec2(current_frame_->features_left_[i]->position_.pt.x,
                     current_frame_->features_left_[i]->position_.pt.y)),
            camera_right_->pixel2camera(
                Vec2(current_frame_->features_right_[i]->position_.pt.x,
                     current_frame_->features_right_[i]->position_.pt.y))};
        Vec3 pworld = Vec3::Zero(); // 存放三角化得到的三维点

        // 4. 三角化，得到三维点坐标，且深度为正
        if (triangulation(poses, points, pworld) && pworld[2] > 0) {
            // 5. 创建新的地图点对象，并设置三维坐标
            auto new_map_point = MapPoint::CreateNewMappoint();
            new_map_point->SetPos(pworld);

            // 6. 将该地图点与左右图的特征点建立观测关系
            new_map_point->AddObservation(current_frame_->features_left_[i]);
            new_map_point->AddObservation(current_frame_->features_right_[i]);

            // 7. 将左右图特征点的 map_point_ 指针指向新建的地图点
            current_frame_->features_left_[i]->map_point_ = new_map_point;
            current_frame_->features_right_[i]->map_point_ = new_map_point;
            cnt_init_landmarks++; // 新增地图点计数

            // 8. 将新地图点加入全局地图
            map_->InsertMapPoint(new_map_point);
        }
    }

    // 9. 将当前帧设置为关键帧，并加入地图
    current_frame_->SetKeyFrame();  // 标记为关键帧
    map_->InsertKeyFrame(current_frame_); // 插入关键帧

    // 10. 通知后端更新地图
    backend_->UpdateMap();

    LOG(INFO) << "Initial map created with " << cnt_init_landmarks
              << " map points";

    return true;
}

bool Frontend::Reset() {
    LOG(INFO) << "Reset is not implemented. ";
    return true;
}

}  // namespace myslam