#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Geometry>


using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {
    // 为什么要归一？ 因为四元数的模长不为1时，表示的旋转就不是纯旋转了，会有缩放等变换的成分，所以需要归一化保证它表示的是一个合法的旋转。
    Quaterniond q1(0.35, 0.2, 0.3, 0.1), q2(-0.5, 0.4, -0.1, 0.2); // 四元数，前面是实部，后面是虚部
    q1.normalize(); // 归一化
    q2.normalize();

    Vector3d t1(0.3, 0.1, 0.1), t2(-0.1, 0.5, 0.3); // 平移向量
    Vector3d p1(0.5, 0, 0.2); // 待变换的点

    Isometry3d T1(q1), T2(q2); // 从旋转矩阵构造变换矩阵
    T1.pretranslate(t1); // 加上平移部分
    T2.pretranslate(t2);

    Vector3d p2 = T2 * T1.inverse() * p1; // 先把p1变换到世界坐标系，再变换到T2的坐标系
    cout << "p1 in T2 frame: " << p2.transpose() << endl;

    return 0;
}