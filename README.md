# slambook2

安装环境
```bash
apt install -y build-essential cmake
apt install -y python3-dev python3-numpy python3-pip
apt install -y libeigen3-dev
apt install -y libopencv-dev
```

安装 opengl
```bash
# ubuntu
# libgl1-mesa-dev        OpenGL 核心开发库（Mesa 实现），提供 libGL.so，用于基础图形渲染
# libglu1-mesa-dev       GLU 工具库，在 OpenGL 之上提供高层功能（如投影、坐标变换等）
# libglfw3-dev           GLFW 窗口与输入库，用于创建窗口、处理键盘鼠标、管理 OpenGL 上下文
# libglew-dev            GLEW 扩展加载库，用于访问现代 OpenGL 的扩展函数（解决函数指针加载问题）
# libglm-dev             GLM 数学库，提供向量/矩阵/变换运算（类似 Eigen，但专为图形设计）
# libglvnd-dev           GLVND 调度层，统一管理不同厂商（NVIDIA/Mesa）的 OpenGL 实现
# libegl1-mesa-dev       EGL 开发库，用于创建 OpenGL 上下文（支持无窗口/嵌入式/Wayland 环境）
# libepoxy-dev           Epoxy 扩展加载库（常用于 GTK/Pango），用于自动加载 OpenGL 函数（替代 GLEW）

sudo apt install -y \
libgl1-mesa-dev \
libglu1-mesa-dev \
libglfw3-dev \
libglew-dev \
libglm-dev \
libglvnd-dev \
libegl1-mesa-dev \
libepoxy-dev
```

添加子模块
```bash
git submodule add https://github.com/stevenlovegrove/Pangolin 3rdparty/Pangolin
git submodule add https://github.com/strasdat/Sophus 3rdparty/Sophus
git submodule add https://github.com/ceres-solver/ceres-solver 3rdparty/ceres-solver
git submodule add https://github.com/RainerKuemmerle/g2o 3rdparty/g2o
git submodule add https://github.com/rmsalinas/DBoW3 3rdparty/DBoW3
git submodule add https://github.com/google/googletest.git 3rdparty/googletest
```

查看子模块状态
```bash
git submodule status
```

安装子模块
```bash
# 安装 Pangolin，参考3rdparty/Pangolin中的readme（先安装依赖）
cd 3rdparty/Pangolin
mkdir build && cd build
cmake ..
make -j
sudo make install
ldconfig
```