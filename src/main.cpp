#include <iostream>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include "Smpl.h"

using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;

const int shape_num = 10;
const int joint_num = 23;
const int cal_joint_free_dim = 24;


int main(int, char**) {

    string modelPath = "model/smpl_female.json";
    string outputPath = "out/vertices.obj";

    // 后期调整为输入的参数
    // 输入的第一个参数为shape，表示在标准模型上的高矮胖瘦偏移量(10, 1)，也就是论文中的beta
    // 第二个参数为pose，表示人体23个关节的相对位姿加上根骨的方向位姿(23+1=24, 3)，也就是论文中的theta
    MatrixXd offset_shape = MatrixXd::Ones(shape_num, 1);
    MatrixXd offset_pose = MatrixXd::Ones(cal_joint_free_dim, 3);
    MatrixXd shape = MatrixXd::Random(shape_num, 1);
    MatrixXd pose = MatrixXd::Random(cal_joint_free_dim, 3);
    shape = 0.03 * 0.5 * (shape + offset_shape);
    pose = 0.2 * 0.5 * (pose + offset_pose);

    // cout << "shape: " << shape << " \n" << "pose: " << pose << endl;

    Smpl* smpl = new Smpl(modelPath, outputPath);
    smpl->loadModel();
    smpl->launch(shape, pose);
    smpl->out();

    return 0;
}
