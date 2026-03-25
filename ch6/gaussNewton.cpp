#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {
  double ar = 1.0, br = 2.0, cr = 1.0;         // 真实参数值（用来生成模拟数据）
  double ae = 2.0, be = -1.0, ce = 5.0;        // 估计参数值（初始猜测值）
  int N = 100;                                 // 数据点（表示生成100个观测点）
  double w_sigma = 1.0;                        // 噪声Sigma值（标准差）
  double inv_sigma = 1.0 / w_sigma;            // Sigma（σ）的倒数，在优化中用于加权误差
  cv::RNG rng;                                 // OpenCV随机数产生器

  vector<double> x_data, y_data;               // 生成带噪声的样本数据
  for (int i = 0; i < N; i++) {
    double x = i / 100.0;
    x_data.push_back(x);
    y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));  // y = exp(ax^2 + bx + c) + w
  }

  // 开始Gauss-Newton迭代，非线性最小二乘拟合
  int iterations = 100;           // 迭代次数
  double cost = 0, lastCost = 0;  // 本次迭代的cost和上一次迭代的cost

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  for (int iter = 0; iter < iterations; iter++) {

    Matrix3d H = Matrix3d::Zero();             // Hessian = J^T W^{-1} J in Gauss-Newton
    Vector3d b = Vector3d::Zero();             // bias
    cost = 0;

    for (int i = 0; i < N; i++) {
      double xi = x_data[i], yi = y_data[i];  // 第i个样本的观测值和预测值
      double error = yi - exp(ae * xi * xi + be * xi + ce); // 计算残差（观测值 - 预测值）
      Vector3d J; // 雅可比矩阵
      J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce);   // de/da
      J[1] = -xi * exp(ae * xi * xi + be * xi + ce);        // de/db
      J[2] = -exp(ae * xi * xi + be * xi + ce);             // de/dc

      H += inv_sigma * inv_sigma * J * J.transpose();
      b += -inv_sigma * inv_sigma * error * J;

      cost += error * error;
    }

    // 求解线性方程 Hx=b
    Vector3d dx = H.ldlt().solve(b);
    if (isnan(dx[0])) {
      cout << "result is nan!" << endl;
      break;
    }

    // 如果cost增加了，说明更新没有改善结果，可能是因为H不够好（比如H不是正定的），因此可以放弃这次更新
    if (iter > 0 && cost >= lastCost) {
      cout << "cost: " << cost << ">= last cost: " << lastCost << ", break." << endl;
      break;
    }

    ae += dx[0];
    be += dx[1];
    ce += dx[2];

    lastCost = cost;

    cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose() <<
         "\t\testimated params: " << ae << "," << be << "," << ce << endl;
  }

  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

  cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;
  return 0;
}
