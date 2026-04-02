#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>
#include <iostream>

#include "common.h"
#include "sophus/se3.hpp"

using namespace Sophus;
using namespace Eigen;
using namespace std;

/**
 * 姿态和内参的结构
 * 1. 姿态：SO3d
 * 2. 平移：Vector3d
 * 3. 焦距：double
 * 4. 径向畸变系数：double, k1, k2
 */
/// 姿态和内参的结构
struct PoseAndIntrinsics {
    PoseAndIntrinsics() {}

    /**
     * 从给定的数据地址初始化姿态和内参
     * @param data_addr 数据地址
     */
    explicit PoseAndIntrinsics(double *data_addr) {
        rotation = SO3d::exp(Vector3d(data_addr[0], data_addr[1], data_addr[2])); // 姿态，指数映射到 SO3d
        translation = Vector3d(data_addr[3], data_addr[4], data_addr[5]); // 平移
        focal = data_addr[6]; // 焦距
        k1 = data_addr[7]; // 径向畸变系数
        k2 = data_addr[8]; // 径向畸变系数
    }

    /**
     * 将估计值放入内存
     * @param data_addr 数据地址
     */
    void set_to(double *data_addr) {
        auto r = rotation.log(); // 对数映射到 tangent space（切空间）
        for (int i = 0; i < 3; ++i) data_addr[i] = r[i];
        for (int i = 0; i < 3; ++i) data_addr[i + 3] = translation[i];
        data_addr[6] = focal; // 焦距
        data_addr[7] = k1; // 径向畸变系数
        data_addr[8] = k2; // 径向畸变系数
    }

    SO3d rotation; // 姿态
    Vector3d translation = Vector3d::Zero(); // 平移
    double focal = 0; // 焦距
    double k1 = 0; // 径向畸变系数
    double k2 = 0; // 径向畸变系数
};

/**
 * 位姿加相机内参的顶点，9维，前三维为so3，接下去为t, f, k1, k2
 */
class VertexPoseAndIntrinsics : public g2o::BaseVertex<9, PoseAndIntrinsics> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     * 构造函数
     */
    VertexPoseAndIntrinsics() {}

    /**
     * 设置为原点
     */
    virtual void setToOriginImpl() override {
        _estimate = PoseAndIntrinsics();
    }

    /**
     * 更新
     * @param update 更新量
     */
    virtual void oplusImpl(const double *update) override {
        _estimate.rotation = SO3d::exp(Vector3d(update[0], update[1], update[2])) * _estimate.rotation; // 姿态，指数映射到 SO3d，然后左乘当前的姿态
        _estimate.translation += Vector3d(update[3], update[4], update[5]); // 平移
        _estimate.focal += update[6]; // 焦距
        _estimate.k1 += update[7]; // 径向畸变系数
        _estimate.k2 += update[8]; // 径向畸变系数
    }

    /**
     * 根据估计值投影一个点
     * @param point 三维点
     * @return 投影点
     */
    Vector2d project(const Vector3d &point) {
        Vector3d pc = _estimate.rotation * point + _estimate.translation; // 投影点
        pc = -pc / pc[2];
        double r2 = pc.squaredNorm(); // 平方范数
        double distortion = 1.0 + r2 * (_estimate.k1 + _estimate.k2 * r2); // 畸变
        return Vector2d(_estimate.focal * distortion * pc[0], // 投影点x
                        _estimate.focal * distortion * pc[1]); // 投影点y
    }

    /**
     * 读取
     * @param in 输入流
     * @return 是否成功
     */
    virtual bool read(istream &in) {return true;}

    /**
     * 写入
     * @param out 输出流
     * @return 是否成功
     */
    virtual bool write(ostream &out) const {return true;}
};

/**
 * 三维点顶点
 */
class VertexPoint : public g2o::BaseVertex<3, Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     * 构造函数
     */
    VertexPoint() {}

    /**
     * 设置为原点
     */
    virtual void setToOriginImpl() override {
        _estimate = Vector3d(0, 0, 0);
    }

    /**
     * 更新
     * @param update 更新量
     */
    virtual void oplusImpl(const double *update) override {
        _estimate += Vector3d(update[0], update[1], update[2]);
    }

    /**
     * 读取
     * @param in 输入流
     * @return 是否成功
     */
    virtual bool read(istream &in) {}

    /**
     * 写入
     * @param out 输出流
     * @return 是否成功
     */
    virtual bool write(ostream &out) const {}
};

/**
 * 投影边
 */
class EdgeProjection :
    public g2o::BaseBinaryEdge<2, Vector2d, VertexPoseAndIntrinsics, VertexPoint> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     * 计算误差
     */
    virtual void computeError() override {
        auto v0 = (VertexPoseAndIntrinsics *) _vertices[0];
        auto v1 = (VertexPoint *) _vertices[1];
        auto proj = v0->project(v1->estimate());
        _error = proj - _measurement;
    }

    /**
     * 读取
     * @param in 输入流
     * @return 是否成功
     */
    virtual bool read(istream &in) {}

    /**
     * 写入
     * @param out 输出流
     * @return 是否成功
     */
    virtual bool write(ostream &out) const {}

};

/**
 * 求解Bundle Adjustment
 * @param bal_problem BALProblem对象
 */
void SolveBA(BALProblem &bal_problem);

int main(int argc, char **argv) {

    if (argc != 2) {
        cout << "usage: bundle_adjustment_g2o bal_data.txt" << endl;
        return 1;
    }

    BALProblem bal_problem(argv[1]);
    bal_problem.Normalize();
    bal_problem.Perturb(0.1, 0.5, 0.5);
    bal_problem.WriteToPLYFile("initial.ply");
    SolveBA(bal_problem);
    bal_problem.WriteToPLYFile("final.ply");

    return 0;
}

/**
 * 求解Bundle Adjustment
 * @param bal_problem BALProblem对象
 */
void SolveBA(BALProblem &bal_problem) {
    const int point_block_size = bal_problem.point_block_size();  // 3D点维度
    const int camera_block_size = bal_problem.camera_block_size();  // 相机参数维度
    double *points = bal_problem.mutable_points();  // 3D点数据
    double *cameras = bal_problem.mutable_cameras();  // 相机参数数据

    // pose dimension 9, landmark is 3
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>> BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    // use LM
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer; // 创建优化器
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    /// build g2o problem
    const double *observations = bal_problem.observations();
    // vertex
    vector<VertexPoseAndIntrinsics *> vertex_pose_intrinsics;
    vector<VertexPoint *> vertex_points;
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        VertexPoseAndIntrinsics *v = new VertexPoseAndIntrinsics();
        double *camera = cameras + camera_block_size * i;
        v->setId(i);
        v->setEstimate(PoseAndIntrinsics(camera));
        optimizer.addVertex(v);
        vertex_pose_intrinsics.push_back(v);
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        VertexPoint *v = new VertexPoint();
        double *point = points + point_block_size * i;
        v->setId(i + bal_problem.num_cameras());
        v->setEstimate(Vector3d(point[0], point[1], point[2]));
        // g2o在BA中需要手动设置待Marg的顶点
        v->setMarginalized(true);
        optimizer.addVertex(v);
        vertex_points.push_back(v);
    }

    // edge
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        EdgeProjection *edge = new EdgeProjection;
        edge->setVertex(0, vertex_pose_intrinsics[bal_problem.camera_index()[i]]);
        edge->setVertex(1, vertex_points[bal_problem.point_index()[i]]);
        edge->setMeasurement(Vector2d(observations[2 * i + 0], observations[2 * i + 1]));
        edge->setInformation(Matrix2d::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(40);

    // set to bal problem
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        double *camera = cameras + camera_block_size * i;
        auto vertex = vertex_pose_intrinsics[i];
        auto estimate = vertex->estimate();
        estimate.set_to(camera);
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        double *point = points + point_block_size * i;
        auto vertex = vertex_points[i];
        for (int k = 0; k < 3; ++k) point[k] = vertex->estimate()[k];
    }
}
