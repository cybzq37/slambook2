#include <iostream>
#include <ceres/ceres.h>
#include "common.h"
#include "SnavelyReprojectionError.h"

using namespace std;

/**
 * 求解Bundle Adjustment
 * @param bal_problem BALProblem对象
 */
void SolveBA(BALProblem &bal_problem);

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "usage: bundle_adjustment_ceres bal_data.txt" << endl;
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
 * 1. 创建ceres::Problem对象
 * 2. 遍历所有观测，创建残差块
 * 3. 添加残差块到ceres::Problem对象
 * 4. 求解ceres::Problem对象
 * 5. 输出求解结果
 * @param bal_problem BALProblem对象
 */
void SolveBA(BALProblem &bal_problem) {
    const int point_block_size = bal_problem.point_block_size();  // 3D点维度
    const int camera_block_size = bal_problem.camera_block_size();  // 相机参数维度
    double *points = bal_problem.mutable_points();  // 3D点数据
    double *cameras = bal_problem.mutable_cameras();  // 相机参数数据

    // Observations is 2 * num_observations long array observations
    // [u_1, u_2, ... u_n], where each u_i is two dimensional, the x
    // and y position of the observation.
    const double *observations = bal_problem.observations(); // 观测数据
    ceres::Problem problem; // 创建ceres::Problem对象

    for (int i = 0; i < bal_problem.num_observations(); ++i) { // 遍历所有观测
        ceres::CostFunction *cost_function; // 创建残差块

        // Each Residual block takes a point and a camera as input
        // and outputs a 2 dimensional Residual
        cost_function = SnavelyReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1]); // 创建残差块

        // If enabled use Huber's loss function.
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0); // 创建损失函数

        // Each observation corresponds to a pair of a camera and a point
        // which are identified by camera_index()[i] and point_index()[i]
        // respectively.
        double *camera = cameras + camera_block_size * bal_problem.camera_index()[i]; // 获取相机参数
        double *point = points + point_block_size * bal_problem.point_index()[i]; // 获取3D点数据

        problem.AddResidualBlock(cost_function, loss_function, camera, point); // 添加残差块到ceres::Problem对象
    }

    // show some information here ...
    std::cout << "bal problem file loaded..." << std::endl;
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
              << bal_problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observations. " << std::endl;

    std::cout << "Solving ceres BA ... " << endl;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
}