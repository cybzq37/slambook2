#ifndef SnavelyReprojection_H
#define SnavelyReprojection_H

#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"

/**
 * 重投影误差
 * @param observation_x 观测的x坐标
 * @param observation_y 观测的y坐标
 */
class SnavelyReprojectionError {
public:
    SnavelyReprojectionError(double observation_x, double observation_y) : observed_x(observation_x),
                                                                           observed_y(observation_y) {}

    /**
     * 重投影误差
     * 1. 将三维点绕相机参数中的角轴旋转
     * 2. 将旋转后的点加上相机参数中的平移
     * 3. 将旋转后的点投影到图像平面
     * 4. 计算投影点与观测点的误差
     * @param camera 相机参数
     * @param point 三维点
     * @param predictions 预测的二维点
     * @return 是否成功
     */
    template<typename T>
    bool operator()(const T *const camera,
                    const T *const point,
                    T *residuals) const {
        // camera[0,1,2] are the angle-axis rotation
        T predictions[2];
        CamProjectionWithDistortion(camera, point, predictions);
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);

        return true;
    }

    /**
     * 将三维点投影到图像平面
     * 1. 将三维点绕相机参数中的角轴旋转，得到旋转后的点
     * 2. 将旋转后的点加上相机参数中的平移，得到平移后的点
     * 3. 将平移后的点除以深度，得到归一化平面上的点
     * 4. 应用径向畸变，得到畸变后的点
     * 5. 将畸变后的点乘以焦距，得到投影点
     * @param camera 相机参数
     * @param point 三维点
     * @param predictions 预测的二维点
     * @return 是否成功
     */
    // camera : 9 dims array
    // [0-2] : angle-axis rotation
    // [3-5] : translateion
    // [6-8] : camera parameter, [6] focal length, [7-8] second and forth order radial distortion
    // point : 3D location.
    // predictions : 2D predictions with center of the image plane.
    template<typename T>
    static inline bool CamProjectionWithDistortion(const T *camera, const T *point, T *predictions) {
        // 将三维点绕相机参数中的角轴旋转
        // Rodrigues' formula
        T p[3];
        AngleAxisRotatePoint(camera, point, p);
        // camera[3,4,5] are the translation
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // Compute the center fo distortion
        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];

        // Apply second and fourth order radial distortion
        const T &l1 = camera[7];
        const T &l2 = camera[8];

        T r2 = xp * xp + yp * yp;
        T distortion = T(1.0) + r2 * (l1 + l2 * r2);

        const T &focal = camera[6];
        predictions[0] = focal * distortion * xp;
        predictions[1] = focal * distortion * yp;

        return true;
    }

    static ceres::CostFunction *Create(const double observed_x, const double observed_y) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
            new SnavelyReprojectionError(observed_x, observed_y)));
    }

private:
    double observed_x;
    double observed_y;
};

#endif // SnavelyReprojection.h

