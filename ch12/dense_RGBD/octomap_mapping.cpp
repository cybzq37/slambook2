/**
 * @file octomap_mapping.cpp
 * @brief RGBD稠密重建程序，将多帧RGBD图像转换为OctoMap八叉树地图
 * 
 * @details 
 * 本程序的主要功能：
 * 1. 读取5帧彩色图像和深度图像，以及对应的相机位姿（位置+四元数表示的旋转）
 * 2. 将每帧深度图像转换为3D点云：
 *    - 使用相机内参（焦距fx/fy、光心cx/cy）进行去畸变投影
 *    - 利用深度值和变换矩阵将像素坐标转换为世界坐标系中的3D点
 * 3. 将所有点云数据插入到OctoMap八叉树结构中：
 *    - 八叉树分辨率为0.01米
 *    - 每帧点云的原点为对应的相机光心位置
 * 4. 更新八叉树的占据信息并保存为二进制文件(octomap.bt)
 * 
 * @return 程序执行成功返回0，文件读取失败返回1
 * 
 * @note 
 * - 深度值为0表示该像素无效（未测量到）
 * - 深度图像的实际深度值 = 图像像素值 / depthScale
 * - 四元数格式为 (w, x, y, z)
 * - 输入文件路径：./data/pose.txt, ./data/color/*.png, ./data/depth/*.png
 * - 输出文件：octomap.bt
 */
#include <iostream>
#include <fstream>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <octomap/octomap.h>    // for octomap 

#include <Eigen/Geometry>
#include <boost/format.hpp>  // for formating strings


int main(int argc, char **argv) {
    vector<cv::Mat> colorImgs, depthImgs;    // 彩色图和深度图
    vector<Eigen::Isometry3d> poses;         // 相机位姿

    ifstream fin("./data/pose.txt");
    if (!fin) {
        cerr << "cannot find pose file" << endl;
        return 1;
    }

    for (int i = 0; i < 5; i++) {
        boost::format fmt("./data/%s/%d.%s"); //图像文件格式
        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "png").str(), -1)); // 使用-1读取原始图像， -1表示不进行任何转换，直接读取图像数据

        double data[7] = {0};
        for (int i = 0; i < 7; i++) {
            fin >> data[i];
        }
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]); // 四元数格式为 (w, x, y, z)
        Eigen::Isometry3d T(q); // 将四元数转换为变换矩阵
        T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2])); // 设置平移部分
        poses.push_back(T);
    }

    // 计算点云并拼接
    // 相机内参 
    double cx = 319.5;
    double cy = 239.5;
    double fx = 481.2;
    double fy = -480.0;
    double depthScale = 5000.0; // 深度图像的缩放因子，实际深度值 = 图像中的值 / depthScale

    cout << "正在将图像转换为 Octomap ..." << endl;

    // octomap tree 
    octomap::OcTree tree(0.01); // 参数为分辨率

    for (int i = 0; i < 5; i++) {
        cout << "转换图像中: " << i + 1 << endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Eigen::Isometry3d T = poses[i];

        octomap::Pointcloud cloud;  // the point cloud in octomap 

        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++) {
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // 深度值
                if (d == 0) continue; // 为0表示没有测量到
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point;
                // 将世界坐标系的点放入点云
                cloud.push_back(pointWorld[0], pointWorld[1], pointWorld[2]);
            }

        // 将点云存入八叉树地图，给定原点，这样可以计算投射线，第二个参数为传感器原点，通常是相机位置
        tree.insertPointCloud(cloud, octomap::point3d(T(0, 3), T(1, 3), T(2, 3)));
    }

    // 更新中间节点的占据信息并写入磁盘
    tree.updateInnerOccupancy();
    cout << "saving octomap ... " << endl;
    tree.writeBinary("octomap.bt");
    return 0;
}
