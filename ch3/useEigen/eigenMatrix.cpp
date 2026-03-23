#include <iostream>
#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

#define MATRIX_SIZE 50

int main(int argc, char **argv)
{
    Matrix<float, 2, 3> matrix_23; // 2行3列的矩阵
    Vector3d v_3d;                   // 3维向量
    Matrix<float, 3, 1> vd_3d;      // 3维向量
    Matrix3d matrix_33 = Matrix3d::Zero(); // 3行3列的矩阵
    matrix_33 << 1, 2, 3, 4, 5, 6, 7, 8, 10;

    Matrix<double, Dynamic, Dynamic> matrix_dynamic; // 动态大小的矩阵
    MatrixXd matrix_x; // 动态大小的矩阵

    matrix_23 << 1, 2, 3, 4, 5, 6;
    cout << "matrix 2x3 from 1 to 6: " << endl
         << matrix_23 << endl;

    for (int i = 0; i < 2;i++)
        for (int j = 0; j < 3;j++)
            cout << matrix_23(i, j) << "\t";
    cout << endl;

    v_3d << 3, 2, 1;
    vd_3d << 4, 5, 6;
    Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d; // 矩阵和向量相乘，结果是2维向量
    cout << "result of matrix_23 * v_3d is " << endl
            << result.transpose() << endl; // transpose()是矩阵转置，输出结果是1行2列

    cout << "trace: " << matrix_33.trace() << endl; // 计算迹
    cout << "sum: " << matrix_33.sum() << endl; // 计算
    cout << "inverse: \n" << matrix_33.inverse() << endl; // 计算逆
    cout << "determinant: " << matrix_33.determinant() << endl; // 计算行列式

    SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33); // 计算特征值和特征向量
    cout << "eigen values = \n" << eigen_solver.eigenvalues() << endl; // 输出特征值
    cout << "eigen vectors = \n" << eigen_solver.eigenvectors() << endl; // 输出特征向量


}