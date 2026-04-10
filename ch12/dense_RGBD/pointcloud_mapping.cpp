#include <iostream>
#include <fstream>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry>
#include <boost/format.hpp>  // for formating strings
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>

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
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "png").str(), -1)); // 使用-1读取原始图像

        double data[7] = {0};
        for (int i = 0; i < 7; i++) {
            fin >> data[i];
        }
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);  // 四元数格式为w,x,y,z
        Eigen::Isometry3d T(q); // 从四元数构造变换矩阵
        T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2])); // 设置平移部分
        poses.push_back(T);
    }

    // 计算点云并拼接
    // 相机内参 
    double cx = 319.5;
    double cy = 239.5;
    double fx = 481.2;
    double fy = -480.0;
    double depthScale = 5000.0;

    cout << "正在将图像转换为点云..." << endl;

    // 定义点云使用的格式：这里用的是XYZRGB
    typedef pcl::PointXYZRGB PointT; // 定义点云类型
    typedef pcl::PointCloud<PointT> PointCloud; // 定义点云类型

    // 新建一个点云
    PointCloud::Ptr pointCloud(new PointCloud); // Ptr为pcl库中智能指针的定义
    for (int i = 0; i < 5; i++) {
        PointCloud::Ptr current(new PointCloud);
        cout << "转换图像中: " << i + 1 << endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Eigen::Isometry3d T = poses[i];
        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++) {
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // 深度值
                if (d == 0) continue; // 为0表示没有测量到
                Eigen::Vector3d point;
                // 将(u,v,d)转成相机坐标系下的点
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                // 将点从相机坐标系转换到世界坐标系
                Eigen::Vector3d pointWorld = T * point;

                PointT p;
                p.x = pointWorld[0];
                p.y = pointWorld[1];
                p.z = pointWorld[2];
                // rgb 颜色信息
                p.b = color.data[v * color.step + u * color.channels()];
                p.g = color.data[v * color.step + u * color.channels() + 1];
                p.r = color.data[v * color.step + u * color.channels() + 2];
                current->points.push_back(p);
            }
        
        // depth filter and statistical removal 
        PointCloud::Ptr tmp(new PointCloud);
        pcl::StatisticalOutlierRemoval<PointT> statistical_filter; // 创建滤波器对象（离群点移除滤波器）
        statistical_filter.setMeanK(50); // 设置在进行统计时考虑查询点最近的50个邻居点
        statistical_filter.setStddevMulThresh(1.0); // 设置标准差乘数阈值，默认值为1.0，表示如果一个点的平均距离与其邻居点的平均距离的差异超过了这个阈值，那么这个点就被认为是离群点
        statistical_filter.setInputCloud(current);
        statistical_filter.filter(*tmp);
        (*pointCloud) += *tmp;
    }

    pointCloud->is_dense = false; // 设置点云为非密集型，允许存在无效点
    cout << "点云共有" << pointCloud->size() << "个点." << endl;

    // voxel filter 
    // 创建一个体素滤波器对象，并设置体素网格的分辨率为0.03米。
    // 体素滤波器会将点云划分为一个个小的立方体（体素），并在每个体素内用一个点来代表该体素内的所有点，从而达到降采样的效果。
    pcl::VoxelGrid<PointT> voxel_filter;
    double resolution = 0.03;
    voxel_filter.setLeafSize(resolution, resolution, resolution);       // resolution
    PointCloud::Ptr tmp(new PointCloud);
    voxel_filter.setInputCloud(pointCloud);
    voxel_filter.filter(*tmp);
    tmp->swap(*pointCloud);

    cout << "滤波之后，点云共有" << pointCloud->size() << "个点." << endl;

    pcl::io::savePCDFileBinary("map.pcd", *pointCloud);
    return 0;
}