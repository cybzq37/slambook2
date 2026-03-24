#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <unistd.h>

using namespace std;
using namespace Eigen;

string trajectory_file = "../trajectory.txt";

void DrawTrajectory(vector<Isometry3d, aligned_allocator<Isometry3d>> trajectory);

int main(int argc, char** argv) {
    vector<Isometry3d, aligned_allocator<Isometry3d>> poses;
    ifstream fin(trajectory_file);
    if(!fin) {
        cerr << "cannot find trajectory file at " << trajectory_file << endl;
        return 1;
    }

    while(!fin.eof()) {
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
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
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
        for(size_t i=0; i<trajectory.size(); i++) {
            Vector3d Ow = trajectory[i].translation(); // 相机中心
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