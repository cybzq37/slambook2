// common.cpp — BALProblem 实现：读入/写出 BAL 文本、PLY、场景归一化与参数扰动。

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "common.h"
#include "rotation.h"
#include "random.h"

typedef Eigen::Map<Eigen::VectorXd> VectorRef;
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;

/// 从文件读一个值；失败则打日志（文件格式非法时后续可能崩溃）
template<typename T>
void FscanfOrDie(FILE *fptr, const char *format, T *value) {
    int num_scanned = fscanf(fptr, format, value);
    if (num_scanned != 1)
        std::cerr << "Invalid UW data file. ";
}

/// 对三维向量各分量叠加 N(0, sigma^2) 噪声。
void PerturbPoint3(const double sigma, double *point) {
    for (int i = 0; i < 3; ++i)
        point[i] += RandNormal() * sigma;
}

/// 取无序数组的中位数（nth_element，原位划分）。
double Median(std::vector<double> *data) {
    int n = data->size();
    std::vector<double>::iterator mid_point = data->begin() + n / 2;
    std::nth_element(data->begin(), mid_point, data->end());
    return *mid_point;
}

/**
 * 构造函数，从文件读取数据，并初始化参数
 * @param filename 文件名
 * @param use_quaternions 是否使用四元数
 */
BALProblem::BALProblem(const std::string &filename, bool use_quaternions) {
    FILE *fptr = fopen(filename.c_str(), "r");

    if (fptr == NULL) {
        std::cerr << "Error: unable to open file " << filename;
        return;
    };

    // 非法文件未做完整校验，读失败时易在后续暴露错误。
    FscanfOrDie(fptr, "%d", &num_cameras_);  // 读取相机数
    FscanfOrDie(fptr, "%d", &num_points_);  // 读取路标点数
    FscanfOrDie(fptr, "%d", &num_observations_);  // 读取观测数

    std::cout << "Header: " << num_cameras_
              << " " << num_points_
              << " " << num_observations_;

    point_index_ = new int[num_observations_];
    camera_index_ = new int[num_observations_];
    observations_ = new double[2 * num_observations_];

    // 计算参数总长度, 9 维相机参数 + 3 维路标点坐标, 共 num_parameters_ 维
    // 每个相机在文件里是 9 个 double（角轴 3 + 平移/内参等 6，BAL 标准约定）。
    // 每个路标点（3D 点）是 3 个 double（(x,y,z)）。
    num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
    parameters_ = new double[num_parameters_];

    // 每条观测：相机 id、点 id、像素 (u, v)
    for (int i = 0; i < num_observations_; ++i) {
        FscanfOrDie(fptr, "%d", camera_index_ + i);
        FscanfOrDie(fptr, "%d", point_index_ + i);
        for (int j = 0; j < 2; ++j) {
            FscanfOrDie(fptr, "%lf", observations_ + 2 * i + j);
        }
    }

    // 文件中依次为所有相机 9 维块，再为所有 3D 点
    for (int i = 0; i < num_parameters_; ++i) {
        FscanfOrDie(fptr, "%lf", parameters_ + i);
    }

    fclose(fptr);

    use_quaternions_ = use_quaternions;
    if (use_quaternions) {
        // 将每相机前 3 维角轴改为 4 维四元数，后 6 维不变；总参数量 10*Nc + 3*Np
        num_parameters_ = 10 * num_cameras_ + 3 * num_points_;
        double *quaternion_parameters = new double[num_parameters_]; // 指向数组的指针，堆上分配
        double *original_cursor = parameters_;
        double *quaternion_cursor = quaternion_parameters;
        for (int i = 0; i < num_cameras_; ++i) {
            AngleAxisToQuaternion(original_cursor, quaternion_cursor);  // 将角轴转换为四元数
            quaternion_cursor += 4;
            original_cursor += 3;
            for (int j = 4; j < 10; ++j) {
                *quaternion_cursor++ = *original_cursor++;
            }
        }
        for (int i = 0; i < 3 * num_points_; ++i) {
            *quaternion_cursor++ = *original_cursor++;
        }
        delete[]parameters_;
        parameters_ = quaternion_parameters;
    }
}

/**
 * 将当前参数写回 BAL 兼容文本（输出时统一写角轴 + 平移形式）
 * 1. 首行四个整数：兼容部分工具；第一、二项均为相机数
 * 2. 每条观测：相机 id、点 id、像素 (u, v)
 * 3. 每相机固定 9 行：角轴(3) + 平移与内参等(6)；四元数模式先转回角轴
 * 4. 每点 3 行：三维坐标 (x, y, z)
 * @param filename 文件名
 */
void BALProblem::WriteToFile(const std::string &filename) const {
    FILE *fptr = fopen(filename.c_str(), "w");

    if (fptr == NULL) {
        std::cerr << "Error: unable to open file " << filename;
        return;
    }

    // 首行四个整数：兼容部分工具；第一、二项均为相机数
    fprintf(fptr, "%d %d %d %d\n", num_cameras_, num_cameras_, num_points_, num_observations_);

    for (int i = 0; i < num_observations_; ++i) {
        fprintf(fptr, "%d %d", camera_index_[i], point_index_[i]);
        for (int j = 0; j < 2; ++j) {
            fprintf(fptr, " %g", observations_[2 * i + j]);
        }
        fprintf(fptr, "\n");
    }

    // 写出时每相机固定 9 行：角轴(3) + 平移与内参等(6)；四元数模式先转回角轴
    for (int i = 0; i < num_cameras(); ++i) {
        double angleaxis[9];
        if (use_quaternions_) {
            QuaternionToAngleAxis(parameters_ + 10 * i, angleaxis);
            memcpy(angleaxis + 3, parameters_ + 10 * i + 4, 6 * sizeof(double));
        } else {
            memcpy(angleaxis, parameters_ + 9 * i, 9 * sizeof(double));
        }
        for (int j = 0; j < 9; ++j) {
            fprintf(fptr, "%.16g\n", angleaxis[j]);
        }
    }

    const double *points = parameters_ + camera_block_size() * num_cameras_;
    for (int i = 0; i < num_points(); ++i) {
        const double *point = points + i * point_block_size();
        for (int j = 0; j < point_block_size(); ++j) {
            fprintf(fptr, "%.16g\n", point[j]);
        }
    }

    fclose(fptr);
}

// 导出 PLY，便于 Meshlab / CloudCompare 查看
void BALProblem::WriteToPLYFile(const std::string &filename) const {
    std::ofstream of(filename.c_str());

    of << "ply"
       << '\n' << "format ascii 1.0"
       << '\n' << "element vertex " << num_cameras_ + num_points_
       << '\n' << "property float x"
       << '\n' << "property float y"
       << '\n' << "property float z"
       << '\n' << "property uchar red"
       << '\n' << "property uchar green"
       << '\n' << "property uchar blue"
       << '\n' << "end_header" << std::endl;

    // 相机光心（由外参推得）——绿色
    double angle_axis[3];
    double center[3];
    for (int i = 0; i < num_cameras(); ++i) {
        const double *camera = cameras() + camera_block_size() * i;
        CameraToAngelAxisAndCenter(camera, angle_axis, center);
        of << center[0] << ' ' << center[1] << ' ' << center[2]
           << " 0 255 0" << '\n';
    }

    // 三维路标——白色
    const double *points = parameters_ + camera_block_size() * num_cameras_;
    for (int i = 0; i < num_points(); ++i) {
        const double *point = points + i * point_block_size();
        for (int j = 0; j < point_block_size(); ++j) {
            of << point[j] << ' ';
        }
        of << " 255 255 255\n";
    }
    of.close();
}

/**
 * 从相机参数块中提取角轴与相机中心
 * @param camera 相机参数块
 * @param angle_axis 角轴
 * @param center 相机中心
 */
void BALProblem::CameraToAngelAxisAndCenter(const double *camera,
                                            double *angle_axis,
                                            double *center) const {
    VectorRef angle_axis_ref(angle_axis, 3);
    if (use_quaternions_) {
        QuaternionToAngleAxis(camera, angle_axis);
    } else {
        angle_axis_ref = ConstVectorRef(camera, 3);
    }

    // 由 -R^T 作用在 camera 块中与平移相关的分量上得到光心 c（等价于 c = -R^T t）
    Eigen::VectorXd inverse_rotation = -angle_axis_ref;
    AngleAxisRotatePoint(inverse_rotation.data(),
                         camera + camera_block_size() - 6,
                         center);
    VectorRef(center, 3) *= -1.0;
}

/**
 * 将角轴与相机中心写回相机参数块
 * @param angle_axis 角轴
 * @param center 相机中心
 * @param camera 相机参数块
 */
void BALProblem::AngleAxisAndCenterToCamera(const double *angle_axis,
                                            const double *center,
                                            double *camera) const {
    ConstVectorRef angle_axis_ref(angle_axis, 3);
    if (use_quaternions_) {
        AngleAxisToQuaternion(angle_axis, camera);
    } else {
        VectorRef(camera, 3) = angle_axis_ref;
    }

    // t = -R * c，写入 camera 块末尾 3 维平移
    AngleAxisRotatePoint(angle_axis, center, camera + camera_block_size() - 6);
    VectorRef(camera + camera_block_size() - 6, 3) *= -1.0;
}

/**
 * 归一化场景，使重建的中位绝对偏差约为 100
 * 1. 对路标点各坐标分量分别取中位数，得到场景“中心”
 * 2. 各点到 median 的 L1 距离的中位数，用作稳健尺度（MAD 思路）
 * 3. 缩放使该 MAD 约为 100，再对点做平移+缩放：X <- scale * (X - median)
 * 4. 对相机参数块中的角轴与相机中心做同样的平移+缩放
 */
void BALProblem::Normalize() {
    // 对路标点各坐标分量分别取中位数，得到场景“中心”
    std::vector<double> tmp(num_points_);
    Eigen::Vector3d median;
    double *points = mutable_points();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < num_points_; ++j) {
            tmp[j] = points[3 * j + i];
        }
        median(i) = Median(&tmp);
    }

    // 各点到 median 的 L1 距离的中位数，用作稳健尺度（MAD 思路）
    for (int i = 0; i < num_points_; ++i) {
        VectorRef point(points + 3 * i, 3);
        tmp[i] = (point - median).lpNorm<1>();
    }

    const double median_absolute_deviation = Median(&tmp);

    // 缩放使该 MAD 约为 100，再对点做平移+缩放：X <- scale * (X - median)
    const double scale = 100.0 / median_absolute_deviation;

    for (int i = 0; i < num_points_; ++i) {
        VectorRef point(points + 3 * i, 3);
        point = scale * (point - median);
    }

    double *cameras = mutable_cameras();
    double angle_axis[3];
    double center[3];
    for (int i = 0; i < num_cameras_; ++i) {
        double *camera = cameras + camera_block_size() * i;
        CameraToAngelAxisAndCenter(camera, angle_axis, center);
        VectorRef(center, 3) = scale * (VectorRef(center, 3) - median);
        AngleAxisAndCenterToCamera(angle_axis, center, camera);
    }
}

/**
 * 对旋转（角轴/四元数经内部转换）、相机平移向量末尾 3 维、路标点分别加入零均值高斯扰动。
 * 1. 对路标点各坐标分量分别加入零均值高斯扰动：X <- X + N(0, point_sigma^2)
 * 2. 对相机参数块中的角轴与相机中心分别加入零均值高斯扰动：
 *    - 旋转在角轴空间扰动：R <- R + N(0, rotation_sigma^2)
 *    - 平移扰动作用在 camera 块末尾 3 维：t <- t + N(0, translation_sigma^2)
 * @param rotation_sigma 旋转扰动标准差
 * @param translation_sigma 平移扰动标准差
 * @param point_sigma 路标点扰动标准差
 */
void BALProblem::Perturb(const double rotation_sigma,
                         const double translation_sigma,
                         const double point_sigma) {
    assert(point_sigma >= 0.0);
    assert(rotation_sigma >= 0.0);
    assert(translation_sigma >= 0.0);

    double *points = mutable_points();
    if (point_sigma > 0) {
        for (int i = 0; i < num_points_; ++i) {
            PerturbPoint3(point_sigma, points + 3 * i);
        }
    }

    // 旋转在角轴空间扰动；平移扰动作用在 camera 块末尾 3 维
    for (int i = 0; i < num_cameras_; ++i) {
        double *camera = mutable_cameras() + camera_block_size() * i;

        double angle_axis[3];
        double center[3];
        CameraToAngelAxisAndCenter(camera, angle_axis, center);
        if (rotation_sigma > 0.0) {
            PerturbPoint3(rotation_sigma, angle_axis);
        }
        AngleAxisAndCenterToCamera(angle_axis, center, camera);

        if (translation_sigma > 0.0)
            PerturbPoint3(translation_sigma, camera + camera_block_size() - 6);
    }
}
