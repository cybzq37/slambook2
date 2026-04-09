#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

#include <boost/timer.hpp>

// for sophus
#include <sophus/se3.hpp>

using Sophus::SE3d;

// for eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

/**********************************************
* 本程序演示了单目相机在已知轨迹下的稠密深度估计
* 使用极线搜索 + NCC 匹配的方式，与书本的 12.2 节对应
* 请注意本程序并不完美，你完全可以改进它——我其实在故意暴露一些问题(这是借口)。
***********************************************/

// ------------------------------------------------------------------
// parameters
const int boarder = 20;         // 边缘宽度
const int width = 640;          // 图像宽度
const int height = 480;         // 图像高度
const double fx = 481.2f;       // 相机内参
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int ncc_window_size = 3;    // NCC 取的窗口半宽度
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); // NCC窗口面积
const double min_cov = 0.1;     // 收敛判定：最小方差
const double max_cov = 10;      // 发散判定：最大方差

// ------------------------------------------------------------------
// 重要的函数
/// 从 REMODE 数据集读取数据
bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    vector<SE3d> &poses,
    cv::Mat &ref_depth
);

/**
 * 根据新的图像更新深度估计
 * @param ref           参考图像
 * @param curr          当前图像
 * @param T_C_R         参考图像到当前图像的位姿
 * @param depth         深度
 * @param depth_cov     深度方差
 * @return              是否成功
 */
bool update(
    const Mat &ref,
    const Mat &curr,
    const SE3d &T_C_R,
    Mat &depth,
    Mat &depth_cov2
);

/**
 * 极线搜索
 * @param ref           参考图像
 * @param curr          当前图像
 * @param T_C_R         位姿
 * @param pt_ref        参考图像中点的位置
 * @param depth_mu      深度均值
 * @param depth_cov     深度方差
 * @param pt_curr       当前点
 * @param epipolar_direction  极线方向
 * @return              是否成功
 */
bool epipolarSearch(
    const Mat &ref,
    const Mat &curr,
    const SE3d &T_C_R,
    const Vector2d &pt_ref,
    const double &depth_mu,
    const double &depth_cov,
    Vector2d &pt_curr,
    Vector2d &epipolar_direction
);

/**
 * 更新深度滤波器
 * @param pt_ref    参考图像点
 * @param pt_curr   当前图像点
 * @param T_C_R     位姿
 * @param epipolar_direction 极线方向
 * @param depth     深度均值
 * @param depth_cov2    深度方向
 * @return          是否成功
 */
bool updateDepthFilter(
    const Vector2d &pt_ref,
    const Vector2d &pt_curr,
    const SE3d &T_C_R,
    const Vector2d &epipolar_direction,
    Mat &depth,
    Mat &depth_cov2
);

/**
 * 计算 NCC 评分
 * @param ref       参考图像
 * @param curr      当前图像
 * @param pt_ref    参考点
 * @param pt_curr   当前点
 * @return          NCC评分
 */
double NCC(const Mat &ref, const Mat &curr, const Vector2d &pt_ref, const Vector2d &pt_curr);

// 双线性灰度插值（在非整数像素坐标（x,y）处，估计灰度值）
inline double getBilinearInterpolatedValue(const Mat &img, const Vector2d &pt) {
    uchar *d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))];  // pt(0, 0)为x, pt(1,0)为y，这里的 img.step 是行跨度（以字节为单位），而不是列数
    double xx = pt(0, 0) - floor(pt(0, 0));  // 小数部分, 点在当前像素格里横向的位置
    double yy = pt(1, 0) - floor(pt(1, 0));  // 小数部分, 点在当前像素格里纵向的位置
    return ((1 - xx) * (1 - yy) * double(d[0]) +
            xx * (1 - yy) * double(d[1]) +
            (1 - xx) * yy * double(d[img.step]) +
            xx * yy * double(d[img.step + 1])) / 255.0;
}

// ------------------------------------------------------------------
// 一些小工具
// 显示估计的深度图
void plotDepth(const Mat &depth_truth, const Mat &depth_estimate);

// 像素坐标转到相机坐标
inline Vector3d px2cam(const Vector2d px) {
    return Vector3d(
        (px(0, 0) - cx) / fx,
        (px(1, 0) - cy) / fy,
        1
    );
}

// 相机坐标转到像素坐标
inline Vector2d cam2px(const Vector3d p_cam) {
    return Vector2d(
        p_cam(0, 0) * fx / p_cam(2, 0) + cx,
        p_cam(1, 0) * fy / p_cam(2, 0) + cy
    );
}

// 检测一个点是否在图像边框内
inline bool inside(const Vector2d &pt) {
    return pt(0, 0) >= boarder && pt(1, 0) >= boarder
           && pt(0, 0) + boarder < width && pt(1, 0) + boarder <= height;
}

// 显示极线匹配
void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr);

// 显示极线
void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr,
                      const Vector2d &px_max_curr);

/// 评测深度估计
void evaludateDepth(const Mat &depth_truth, const Mat &depth_estimate);
// ------------------------------------------------------------------


int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: dense_mapping path_to_test_dataset" << endl;
        return -1;
    }

    // 从数据集读取数据
    vector<string> color_image_files;
    vector<SE3d> poses_TWC;
    Mat ref_depth;
    // TWC表示相机到世界坐标系，TCW表示世界到相机坐标系
    bool ret = readDatasetFiles(argv[1], color_image_files, poses_TWC, ref_depth); 
    if (ret == false) {
        cout << "Reading image files failed!" << endl;
        return -1;
    }
    cout << "read total " << color_image_files.size() << " files." << endl;

    // 第一张图
    Mat ref = imread(color_image_files[0], 0);  // 0表示以灰度方式读取
    SE3d pose_ref_TWC = poses_TWC[0];    // 第一张图的位姿（相机坐标）
    double init_depth = 3.0;    // 深度初始值
    double init_cov2 = 3.0;     // 方差初始值
    Mat depth(height, width, CV_64F, init_depth);       // 深度图
    Mat depth_cov2(height, width, CV_64F, init_cov2);   // 深度图方差（初始化为3.0）

    for (int index = 1; index < color_image_files.size(); index++) {  // 从第二张图开始
        cout << "*** loop " << index << " ***" << endl;
        Mat curr = imread(color_image_files[index], 0); // 0表示以灰度方式读取
        if (curr.data == nullptr) continue;
        SE3d pose_curr_TWC = poses_TWC[index];  // 获取 TWC 位姿信息
        SE3d pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC;   // 坐标转换关系： T_C_W * T_W_R = T_C_R，把第一帧中的点投影到当前帧，ref坐标系-》世界-》curr坐标系
        update(ref, curr, pose_T_C_R, depth, depth_cov2); // 更新深度图，depth 和 depth_cov2 默认都是初始值，都会被更新
        evaludateDepth(ref_depth, depth);  // 评测深度估计的结果
        plotDepth(ref_depth, depth); // 显示深度图
        imshow("image", curr);
        waitKey(1);
    }

    cout << "estimation returns, saving depth map ..." << endl;
    imwrite("depth.png", depth);
    cout << "done." << endl;

    return 0;
}

bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    std::vector<SE3d> &poses,
    cv::Mat &ref_depth) {
    ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
    if (!fin) return false;

    while (!fin.eof()) {
        // 数据格式：图像文件名 tx, ty, tz, qx, qy, qz, qw （注意：是 TWC 而非 TCW）  tx ty tz 是平移，qx qy qz qw 是四元数
        string image;
        fin >> image;
        double data[7];
        for (double &d:data) fin >> d;

        color_image_files.push_back(path + string("/images/") + image);
        poses.push_back(
            SE3d(Quaterniond(data[6], data[3], data[4], data[5]),
                 Vector3d(data[0], data[1], data[2]))
        );
        if (!fin.good()) break;
    }
    fin.close();

    // load reference depth
    fin.open(path + "/depthmaps/scene_000.depth"); // 一个像素一个深度，以厘米为单位
    ref_depth = cv::Mat(height, width, CV_64F); // 64位浮点数
    if (!fin) return false;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            double depth = 0;
            fin >> depth; // 从文件中读取深度值
            ref_depth.ptr<double>(y)[x] = depth / 100.0; // 数据集中的深度值以厘米为单位，转换为米
        }

    return true;
}

// 对整个深度图进行更新
/**
 * @brief 深度图更新函数 - 使用极线搜索和高斯融合进行密集单目深度估计
 * 在单目视觉中，深度无法直接从单张图像中获得，但通过多帧图像之间的几何关系（例如相机的位姿变化），可以间接估计深度。
 * 已知信息：
 * - 参考帧图像（ref）和当前帧图像（curr）
 * - 参考帧到当前帧的位姿变换（T_C_R）
 * - 当前的深度估计（depth）和深度不确定性（depth_cov2）
 * 
 * 算法流程：
 * 1. 遍历参考图像中去除边缘后的所有像素点
 * 2. 收敛性检查：跳过已收敛（方差<min_cov）或已发散（方差>max_cov）的像素
 * 3. 极线搜索：在当前帧中沿极线搜索参考帧像素的匹配点
 *    - 利用当前深度估计的均值和标准差确定搜索范围
 *    - 使用NCC（标准化交叉相关）进行相似度匹配
 *    - 输出匹配点坐标和极线方向向量
 * 4. 匹配验证：若搜索失败则跳过该像素
 * 5. 深度滤波更新：基于成功匹配的点对进行高斯融合
 *    - 利用极线方向计算深度的观测值和不确定性
 *    - 更新深度均值和方差，使其逐步收敛到真实值
 * 
 * @param ref 参考帧图像
 * @param curr 当前帧图像
 * @param T_C_R 参考帧到当前帧的位姿变换 (SE3)
 * @param depth 深度图均值 (输入输出)
 * @param depth_cov2 深度图方差 (输入输出)
 * 
 * @return true 深度图更新成功
 */
bool update(const Mat &ref, const Mat &curr, const SE3d &T_C_R, Mat &depth, Mat &depth_cov2) {
    // 遍历参考图像中去除边缘后的所有像素点
    for (int x = boarder; x < width - boarder; x++)  // boarder为边缘宽度
        for (int y = boarder; y < height - boarder; y++) {
            // 检查深度收敛性：方差过小表示已收敛，方差过大表示已发散，都跳过
            // min_cov = 0.1（收敛判定阈值），max_cov = 10（发散判定阈值）
            if (depth_cov2.ptr<double>(y)[x] < min_cov || depth_cov2.ptr<double>(y)[x] > max_cov)
                continue;
            
            // 在当前帧中沿极线搜索该像素的匹配点
            Vector2d pt_curr;                    // 当前帧中匹配点的位置
            Vector2d epipolar_direction;         // 极线方向（单位向量），用于后续计算深度不确定性
            
            bool ret = epipolarSearch(
                ref,                             // 参考帧
                curr,                            // 当前帧
                T_C_R,                           // 参考帧到当前帧的位姿变换
                Vector2d(x, y),                  // 参考帧中该像素的坐标
                depth.ptr<double>(y)[x],         // 当前深度估计的均值
                sqrt(depth_cov2.ptr<double>(y)[x]),  // 当前深度估计的标准差（方差的平方根）
                pt_curr,                         // 输出：匹配点在当前帧中的坐标
                epipolar_direction               // 输出：极线方向
            );

            // 如果极线搜索失败（未找到满足NCC阈值的匹配点），则跳过该像素
            if (ret == false)
                continue;

            // 匹配成功，使用该匹配点更新深度滤波器（高斯融合）
            updateDepthFilter(
                Vector2d(x, y),      // 参考帧中的点
                pt_curr,             // 当前帧中的匹配点
                T_C_R,               // 位姿
                epipolar_direction,  // 极线方向
                depth,               // 深度均值（会被更新）
                depth_cov2           // 深度方差（会被更新）
            );
        }
    return true;  // 更新完成
}

// 极线搜索
// 方法见书 12.2 12.3 两节
bool epipolarSearch(
    const Mat &ref, // 参考帧
    const Mat &curr, // 当前帧
    const SE3d &T_C_R, // 参考帧到当前帧的位姿
    const Vector2d &pt_ref,  // 参考帧中点的位置
    const double &depth_mu,  // 深度均值
    const double &depth_cov, // 深度方差
    Vector2d &pt_curr, // 当前帧中匹配点的位置（输出参数）
    Vector2d &epipolar_direction // 当前帧中的极线方向（单位向量），用于后续更新深度滤波器（输出参数）
) {
    Vector3d f_ref = px2cam(pt_ref); // 参考帧中的点位置，转为相机坐标系
    f_ref.normalize(); // 归一化为单位向量
    Vector3d P_ref = f_ref * depth_mu;    // 参考帧的 P 向量，方向 x 深度 = 三维坐标点

    Vector2d px_mean_curr = cam2px(T_C_R * P_ref); // 参考帧中的点，经过位姿变换后在当前帧中的像素位置（深度均值点投影的位置）
    // 计算深度搜索范围的下界
    // 使用均值减去3倍标准差，确保搜索范围覆盖大约99.7%的正态分布数据
    // 这样可以排除过度偏离均值的异常深度值，提高匹配的稳定性和准确性
    // 初始都为 3.0，随着迭代更新，深度估计会逐渐收敛，方差会逐渐减小，搜索范围也会逐渐缩小，从而提高匹配的效率和准确性
    // d_min = depth_mu - 3 * depth_cov
    double d_min = depth_mu - 3 * depth_cov, d_max = depth_mu + 3 * depth_cov;  // 深度均值的 3 倍方差范围内搜索
    if (d_min < 0.1) d_min = 0.1;  // 深度不可能为负数，且小于0.1m的点我们不考虑了
    Vector2d px_min_curr = cam2px(T_C_R * (f_ref * d_min));    // 当前帧，按最小深度投影的像素
    Vector2d px_max_curr = cam2px(T_C_R * (f_ref * d_max));    // 当前帧，按最大深度投影的像素

    Vector2d epipolar_line = px_max_curr - px_min_curr;    // 当前帧中的极线（线段形式）
    epipolar_direction = epipolar_line;        // 极线方向
    epipolar_direction.normalize();
    double half_length = 0.5 * epipolar_line.norm();  // 极线线段的半长度
    if (half_length > 100) half_length = 100;  // 我们不希望搜索太多东西

    // 取消此句注释以显示极线（线段）
    // showEpipolarLine( ref, curr, pt_ref, px_min_curr, px_max_curr );

    // 在极线上搜索，以深度均值点为中心，左右各取半长度（块匹配）
    double best_ncc = -1.0;
    Vector2d best_px_curr; // 最佳匹配点
    for (double l = -half_length; l <= half_length; l += 0.7) { // l+=sqrt(2) 是为了保证在极线上每隔一个像素就进行一次匹配，0.7是经验值，可以根据实际情况调整
        Vector2d px_curr = px_mean_curr + l * epipolar_direction;  // 待匹配点
        if (!inside(px_curr)) // 如果待匹配点不在图像边界内，则跳过
            continue;
        // 计算待匹配点与参考帧的 NCC
        double ncc = NCC(ref, curr, pt_ref, px_curr);
        if (ncc > best_ncc) {
            best_ncc = ncc;
            best_px_curr = px_curr;
        }
    }
    if (best_ncc < 0.85f)  // 只相信 NCC 很高的匹配
        return false;
    pt_curr = best_px_curr;
    return true;
}


/**
 * @brief 计算两帧之间的零均值归一化互相关 (Zero-Mean Normalized Cross-Correlation)
 * 
 * NCC算法用于衡量两个图像块之间的相似度,常用于立体匹配、光流估计等计算机视觉任务中。
 * 通过计算零均值的互相关系数,可以消除光照变化的影响,提高匹配的鲁棒性。
 * 
 * @param ref 参考帧图像 (灰度图,8位单通道)
 * @param curr 当前帧图像 (灰度图,8位单通道)
 * @param pt_ref 在参考帧中要匹配的点的坐标 (floating-point precision)
 * @param pt_curr 在当前帧中的对应点坐标 (浮点坐标,支持双线性插值)
 * 
 * @return double 相关系数,范围为[-1, 1]。值越接近1表示两个图像块越相似;
 *                值越接近-1表示两个图像块越相反;值接近0表示无相关性
 * 
 * @algorithm
 *   1. 在参考帧和当前帧中分别提取以给定点为中心的ncc_window_size大小的图像块
 *   2. 计算两个图像块的平均灰度值
 *   3. 对所有像素进行零均值处理
 *   4. 计算零均值互相关系数: NCC = Σ(ref_i - mean_ref)(curr_i - mean_curr) 
 *                                    / √[Σ(ref_i - mean_ref)² * Σ(curr_i - mean_curr)²]
 *
 * @brief 零均值处理 (Zero-Mean Normalization)
 * 
 * 零均值处理是指将数据中的每个值减去该组数据的平均值,使得处理后的数据均值为0。
 * 这是归一化的一种常见方法,也称为中心化(Centering)。
 * 
 * @purpose 零均值处理的主要作用:
 *   1. 消除光照变化的影响 - 在图像匹配中,不同光照条件下的同一物体会有不同的灰度值,
 *      通过零均值处理可以使匹配对光照变化具有鲁棒性
 *   2. 突出图像细节 - 减去均值后只保留相对变化信息,增强了纹理特征
 *   3. 提高相关系数的可靠性 - 使得相关系数只反映两个数据序列的相似形状,
 *      而不受绝对亮度值的影响
 * 
 * @note 当前帧的像素值通过双线性插值获取,以支持亚像素级精度的匹配
 */
double NCC(
    const Mat &ref,  // 参考帧
    const Mat &curr, // 当前帧
    const Vector2d &pt_ref, // 参考帧中点的位置
    const Vector2d &pt_curr // 当前帧中点的位置
) {
    // 零均值-归一化互相关
    // 先算均值
    double mean_ref = 0, mean_curr = 0;
    vector<double> values_ref, values_curr; // 参考帧和当前帧的均值
    for (int x = -ncc_window_size; x <= ncc_window_size; x++)
        for (int y = -ncc_window_size; y <= ncc_window_size; y++) {
            double value_ref = double(ref.ptr<uchar>(int(y + pt_ref(1, 0)))[int(x + pt_ref(0, 0))]) / 255.0;
            mean_ref += value_ref;

            double value_curr = getBilinearInterpolatedValue(curr, pt_curr + Vector2d(x, y)); // 当前帧的值需要双线性插值
            mean_curr += value_curr;

            values_ref.push_back(value_ref);
            values_curr.push_back(value_curr);
        }

    mean_ref /= ncc_area; // 参考帧的灰度值总和除以窗口面积，得到平均灰度值
    mean_curr /= ncc_area; // 当前帧的灰度值总和除以窗口面积，得到平均灰度值

    // 计算 Zero mean NCC
    double numerator = 0, demoniator1 = 0, demoniator2 = 0; // 分子和分母的两个部分
    for (int i = 0; i < values_ref.size(); i++) {
        double n = (values_ref[i] - mean_ref) * (values_curr[i] - mean_curr);
        numerator += n;
        demoniator1 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref);
        demoniator2 += (values_curr[i] - mean_curr) * (values_curr[i] - mean_curr);
    }
    return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);   // 防止分母出现零
}

bool updateDepthFilter(
    const Vector2d &pt_ref,           // 参考帧中的点坐标
    const Vector2d &pt_curr,          // 当前帧中的点坐标
    const SE3d &T_C_R,                // 参考帧到当前帧的位姿变换
    const Vector2d &epipolar_direction, // 极线方向（单位向量）
    Mat &depth,                       // 深度图（输入输出）
    Mat &depth_cov2) {                // 深度方差图（输入输出）
    
    // 获取当前帧到参考帧的位姿变换（T_C_R的逆）
    SE3d T_R_C = T_C_R.inverse();
    
    // 将参考帧和当前帧的像素坐标转换为归一化相机坐标
    Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();  // 归一化为单位向量（深度为 1，相当于方向向量）
    Vector3d f_curr = px2cam(pt_curr);
    f_curr.normalize(); // 归一化为单位向量

    // 使用三角化计算深度
    // 基本方程：d_ref * f_ref = d_cur * (R_RC * f_cur) + t_RC
    // 其中 f2 = R_RC * f_cur
    // 转化为线性方程组：
    // [ f_ref^T f_ref,  -f_ref^T f2   ] [d_ref]   [f_ref^T t]
    // [ f2^T f_ref,     -f2^T f2      ] [d_cur] = [f2^T t   ]
    
    Vector3d t = T_R_C.translation();  // 参考帧到当前帧的平移向量
    Vector3d f2 = T_R_C.so3() * f_curr;  // 当前帧的单位方向向量旋转到参考帧坐标系
    Vector2d b = Vector2d(t.dot(f_ref), t.dot(f2));  // 方程组右侧向量
    
    // 构造方程组的系数矩阵A
    Matrix2d A;
    A(0, 0) = f_ref.dot(f_ref);
    A(0, 1) = -f_ref.dot(f2);
    A(1, 0) = -A(0, 1);
    A(1, 1) = -f2.dot(f2);
    
    // 求解线性方程组，得到两帧中的深度值
    Vector2d ans = A.inverse() * b;
    
    Vector3d xm = ans[0] * f_ref;      // 参考帧中的三维点
    Vector3d xn = t + ans[1] * f2;     // 当前帧中的三维点
    Vector3d p_esti = (xm + xn) / 2.0; // 取两个结果的平均作为最终的三维点估计
    double depth_estimation = p_esti.norm();  // 计算深度值（点到相机原点的距离）

    // 计算深度的不确定性（考虑一个像素的误差）
    Vector3d p = f_ref * depth_estimation;
    Vector3d a = p - t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    
    // 计算三角形的各个角度
    double alpha = acos(f_ref.dot(t) / t_norm);  // 参考帧处的角度
    double beta = acos(-a.dot(t) / (a_norm * t_norm));  // 当前帧处的角度
    
    // 在当前帧中沿着极线方向移动一个像素
    Vector3d f_curr_prime = px2cam(pt_curr + epipolar_direction);
    f_curr_prime.normalize();
    double beta_prime = acos(f_curr_prime.dot(-t) / t_norm);  // 移动后的角度
    
    double gamma = M_PI - alpha - beta_prime;  // 三角形第三个角
    double p_prime = t_norm * sin(beta_prime) / sin(gamma);  // 由三角形正弦定理计算的新深度
    double d_cov = p_prime - depth_estimation;  // 深度误差
    double d_cov2 = d_cov * d_cov;  // 深度的方差（误差的平方）

    // 高斯融合：将新观测与之前的估计进行融合
    double mu = depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];  // 之前的深度均值
    double sigma2 = depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];  // 之前的深度方差

    // 计算融合后的深度均值
    double mu_fuse = (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2);
    // 计算融合后的深度方差
    double sigma_fuse2 = (sigma2 * d_cov2) / (sigma2 + d_cov2);

    // 更新深度和方差
    depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = mu_fuse;
    depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = sigma_fuse2;

    return true;
}

// 后面这些太简单我就不注释了（其实是因为懒）
void plotDepth(const Mat &depth_truth, const Mat &depth_estimate) {
    imshow("depth_truth", depth_truth * 0.4);
    imshow("depth_estimate", depth_estimate * 0.4);
    imshow("depth_error", depth_truth - depth_estimate);
    waitKey(1);
}

void evaludateDepth(const Mat &depth_truth, const Mat &depth_estimate) {
    double ave_depth_error = 0;     // 平均误差
    double ave_depth_error_sq = 0;      // 平方误差
    int cnt_depth_data = 0;
    for (int y = boarder; y < depth_truth.rows - boarder; y++)
        for (int x = boarder; x < depth_truth.cols - boarder; x++) {
            double error = depth_truth.ptr<double>(y)[x] - depth_estimate.ptr<double>(y)[x];
            ave_depth_error += error;
            ave_depth_error_sq += error * error;
            cnt_depth_data++;
        }
    ave_depth_error /= cnt_depth_data;
    ave_depth_error_sq /= cnt_depth_data;

    cout << "Average squared error = " << ave_depth_error_sq << ", average error: " << ave_depth_error << endl;
}

void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr) {
    Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, cv::COLOR_GRAY2BGR);
    cv::cvtColor(curr, curr_show, cv::COLOR_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 0, 250), 2);
    cv::circle(curr_show, cv::Point2f(px_curr(0, 0), px_curr(1, 0)), 5, cv::Scalar(0, 0, 250), 2);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1);
}

void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr,
                      const Vector2d &px_max_curr) {

    Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, cv::COLOR_GRAY2BGR);
    cv::cvtColor(curr, curr_show, cv::COLOR_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::line(curr_show, Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), Point2f(px_max_curr(0, 0), px_max_curr(1, 0)),
             Scalar(0, 255, 0), 1);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1);
}
