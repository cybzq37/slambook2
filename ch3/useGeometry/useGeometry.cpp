#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {
    Matrix3d rotation_matrix = Matrix3d::Identity(); // 旋转矩阵, 初始为单位矩阵
    AngleAxisd rotation_vector(M_PI / 4, Vector3d(0, 0, 1)); // 绕 z 轴旋转 45 度
    rotation_matrix = rotation_vector.toRotationMatrix(); // 将旋转向量转换为旋转矩阵
    cout.precision(3); // 输出精度为小数点后三位
    cout << "旋转矩阵 = \n"
         << rotation_vector.matrix() << endl;

    Vector3d v(1, 0, 0); // 定义一个向量
    Vector3d v_rotated = rotation_vector * v; // 旋转向量
    cout << "旋转后的向量 = \n" << v_rotated.transpose() << endl;

    v_rotated = rotation_matrix * v; // 使用旋转矩阵旋转向量
    cout << "旋转后的向量 = \n" << v_rotated.transpose() << endl;

    Vector3d euler_angles = rotation_vector.matrix().eulerAngles(2, 1, 0); // 提取欧拉角，顺序为 ZYX, 即roll pitch yaw
    cout << "欧拉角 = \n" << euler_angles.transpose() << endl;

    // 欧式矩阵
    Isometry3d T = Isometry3d::Identity(); // 定义一个欧式矩阵
    T.rotate(rotation_vector); // 设置旋转部分
    T.pretranslate(Vector3d(1, 3, 4)); // 设置平移部分
    cout << "欧式矩阵 = \n" << T.matrix() << endl;

    Vector3d v_transformed = T * v; // 使用欧式矩阵变换向量
    cout << "变换后的向量 = \n" << v_transformed.transpose() << endl;

    // 仿射矩阵
    Affine3d T_affine = Affine3d::Identity(); // 定义一个仿射矩阵
    T_affine.rotate(rotation_vector); // 设置旋转部分
    T_affine.pretranslate(Vector3d(1, 3, 4)); // 设置平移部分
    T_affine.scale(2.0); // 设置缩放部分
    cout << "仿射矩阵 = \n" << T_affine.matrix() << endl;

    // 射影变换
    Projective3d T_projective = Projective3d::Identity(); // 定义一个射影矩阵
    T_projective.rotate(rotation_vector); // 设置旋转部分
    T_projective.pretranslate(Vector3d(1, 3, 4)); // 设置平移部分
    T_projective.scale(2.0); // 设置缩放部分
    T_projective(3,0) = 0.1; // 设置射影部分
    cout << "射影矩阵 = \n" << T_projective.matrix() << endl;

    // 四元数
    Quaterniond q(rotation_vector); // 将旋转向量转换为四元数
    cout << "四元数 = \n" << q.coeffs().transpose() << endl; // 注意 Eigen 中四元数的 coeffs() 函数返回的是 (x, y, z, w) 的顺序
    q = Quaterniond(rotation_matrix); // 将旋转矩阵转换为四元数
    cout << "四元数 = \n" << q.coeffs().transpose() << endl;
    v_rotated = q * v; // 使用四元数旋转向量, 数学上是 q * v * q^{-1}, 但 Eigen 已经重载了运算符
    cout << "旋转后的向量 = \n" << v_rotated.transpose() << endl;
    // 使用常规算法， 含义是将向量 v 看作纯四元数 (0, v_x, v_y, v_z)，然后进行旋转
    cout << "should be equal to " << (q * Quaterniond(0, 1, 0, 0) * q.inverse()).coeffs().transpose() << endl;


    return 0;
}