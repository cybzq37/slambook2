#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <unistd.h>

using namespace std;
using namespace Eigen;

string trajectory_file = "../trajectory.txt";

void DrawTrajectory(vector<Isometry3d, aligned_allocator<Isometry3d>> trajectory);

int main(int argc, char** argv) {
    // vector<T, Allocator> T 表示类型，Allocator表示分配器，内存管理策略
    vector<Isometry3d, aligned_allocator<Isometry3d>> poses; 
    ifstream fin(trajectory_file);
    if(!fin) {
        cerr << "cannot find trajectory file at " << trajectory_file << endl;
        return 1;
    }

    while(!fin.eof()) {
        // t: translation 平移 q: quaternion 四元数
        // tx ty tz 相机位置，qx qy qz qw 相机姿态四元数
        double time, tx,ty,tz,qx,qy,qz,qw;
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        Isometry3d Twr(Quaterniond(qw, qx, qy, qz));
        Twr.pretranslate(Vector3d(tx, ty, tz)); // 将平移部分添加到变换矩阵中
        poses.push_back(Twr);
    }

    cout << "read total " << poses.size() << " pose entries" << endl;

    DrawTrajectory(poses);

    return 0;
}

void DrawTrajectory(vector<Isometry3d, aligned_allocator<Isometry3d>> trajectory) {
    if(trajectory.empty()) {
        cerr << "trajectory is empty!" << endl;
        return;
    }

    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST); // 开启深度测试，确保正确显示3D场景中的前后关系
    glEnable(GL_BLEND);      // 开启混合功能，允许透明度效果
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // 设置混合函数，使用源颜色的alpha值进行混合

    // 设置相机的投影矩阵和模型视图矩阵（这里的相机指的是渲染画面的相机）
    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(  // 相机内参
                1024, 768,  // 图像宽高
                500, 500,   // fx, fy（焦距）
                512, 389,   // cx, cy（主点）
                0.1, 1000   // 近裁剪面，远裁剪面
            ),  
            pangolin::ModelViewLookAt(    // 相机外参
                0, -0.1, -1.8,   // 相机位置
                0, 0, 0,         // 看向哪里
                0.0, -1.0, 0.0   // 上方向
            ) 
    );

    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(&handler);

    while(pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f,1.0f,1.0f,1.0f);

        // draw trajectory
        glLineWidth(2);
        for(size_t i=0; i<trajectory.size(); i++) {    // trajectory[i] 欧式变换（包含了相机的位置和姿态）
            Vector3d Ow = trajectory[i].translation(); // 取出相机的中心位置，不包含姿态（世界坐标）
            // 绘制相机坐标系的三个轴，长度为0.1,Vector3d是相机坐标，Xw Yw Zw 是世界坐标
            Vector3d Xw = trajectory[i] * (0.1 * Vector3d(1,0,0)); // 相机坐标系的x轴
            Vector3d Yw = trajectory[i] * (0.1 * Vector3d(0,1,0)); // 相机坐标系的y轴
            Vector3d Zw = trajectory[i] * (0.1 * Vector3d(0,0,1)); // 相机坐标系的z轴
            glBegin(GL_LINES);
            glColor3f(1.0f,0.0f,0.0f); // 红色表示x轴
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Xw[0], Xw[1], Xw[2]);
            glColor3f(0.0f,1.0f,0.0f); // 绿色表示y轴
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Yw[0], Yw[1], Yw[2]);
            glColor3f(0.0f,0.0f,1.0f); // 蓝色表示z轴
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Zw[0], Zw[1], Zw[2]);
            glEnd();
        }

        glColor3f(0.0f, 0.0f, 0.0f); // 黑色表示轨迹线
        glBegin(GL_LINE_STRIP);
        for (size_t i = 0; i < trajectory.size(); i++) {
            auto p = trajectory[i].translation();
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();

        pangolin::FinishFrame();
        usleep(50000);   // sleep 50 ms
    }
}