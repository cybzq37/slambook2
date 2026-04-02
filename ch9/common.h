#pragma once  // 防止重复包含

/// 封装 Bundle Adjustment in the Large (BAL) 数据集：从文本读入观测与初值，
/// 提供读写、归一化与加噪，并为 Ceres 等优化器暴露扁平参数块布局。
class BALProblem {
public:
    /// 从 BAL 文本文件加载：前两行头为相机数、路标点数、观测数；随后每行一条观测
    /// （相机索引、点索引、像素 u/v）；最后是所有相机参数与 3D 点坐标。
    /// \param use_quaternions 若 true，加载后将旋转从角轴转为四元数，相机块由 9 维变为 10 维。
    explicit BALProblem(const std::string &filename, bool use_quaternions = false);

    ~BALProblem() {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
    }

    /// 将当前参数写回 BAL 兼容文本（输出时统一写角轴 + 平移形式）。
    void WriteToFile(const std::string &filename) const;

    /// 导出 PLY：相机光心为绿点，路标点为白点，便于 Meshlab / CloudCompare 查看。
    void WriteToPLYFile(const std::string &filename) const;

    /// 以路标点的边际中位数与尺度归一化场景，使重建的中位绝对偏差约为 100。
    void Normalize();

    /// 对旋转（角轴/四元数经内部转换）、相机平移向量末尾 3 维、路标点分别加入零均值高斯扰动。
    void Perturb(const double rotation_sigma,
                 const double translation_sigma,
                 const double point_sigma);

    /// 单目相机参数块长度：角轴(3)+t 与内参等共 9 维，或四元数(4)+后续 6 维共 10 维。
    int camera_block_size() const { return use_quaternions_ ? 10 : 9; }

    /// 单个 3D 点维度（x, y, z）。
    int point_block_size() const { return 3; }

    int num_cameras() const { return num_cameras_; }

    int num_points() const { return num_points_; }

    int num_observations() const { return num_observations_; }

    /// 扁平参数总长 = camera_block_size() * num_cameras_ + 3 * num_points_。
    int num_parameters() const { return num_parameters_; }

    const int *point_index() const { return point_index_; }

    const int *camera_index() const { return camera_index_; }

    /// 观测像素坐标，按观测展开：第 i 条为 [u, v] 即 observations_[2*i], observations_[2*i+1]。
    const double *observations() const { return observations_; }

    /// 全部优化变量连续存储：前段为所有相机块，后段为所有路标点。
    const double *parameters() const { return parameters_; }

    const double *cameras() const { return parameters_; }

    const double *points() const { return parameters_ + camera_block_size() * num_cameras_; }

    /// 可写相机参数区起始地址（与 cameras() 同一区域）。
    double *mutable_cameras() { return parameters_; }

    double *mutable_points() { return parameters_ + camera_block_size() * num_cameras_; }

    /// 第 i 条观测所对应相机参数块的起始可写指针。
    double *mutable_camera_for_observation(int i) {
        return mutable_cameras() + camera_index_[i] * camera_block_size();
    }

    /// 第 i 条观测所对应 3D 点的起始可写指针。
    double *mutable_point_for_observation(int i) {
        return mutable_points() + point_index_[i] * point_block_size();
    }

    const double *camera_for_observation(int i) const {
        return cameras() + camera_index_[i] * camera_block_size();
    }

    const double *point_for_observation(int i) const {
        return points() + point_index_[i] * point_block_size();
    }

private:
    /// 从内部 camera 块解析角轴（或四元数转轴角）、并依据 c = -R^T t 计算相机中心。
    void CameraToAngelAxisAndCenter(const double *camera,
                                    double *angle_axis,
                                    double *center) const;

    /// 由角轴与相机中心写回 camera 块（含 t = -R c 及四元数/角轴编码）。
    void AngleAxisAndCenterToCamera(const double *angle_axis,
                                    const double *center,
                                    double *camera) const;

    int num_cameras_;
    int num_points_;
    int num_observations_;
    int num_parameters_;
    /// 相机旋转是否用四元数存储（影响 camera_block_size）。
    bool use_quaternions_;

    int *point_index_;   ///< 第 k 条观测对应的路标点在 points 中的索引
    int *camera_index_;  ///< 第 k 条观测对应的相机在 cameras 中的索引
    double *observations_;  ///< 所有观测的像素坐标，长度 2 * num_observations_
    double *parameters_;    ///< 相机块 + 点块的大数组，由本类分配与释放
};
