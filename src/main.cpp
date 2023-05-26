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

    // ���ڵ���Ϊ����Ĳ���
    // ����ĵ�һ������Ϊshape����ʾ�ڱ�׼ģ���ϵĸ߰�����ƫ����(10, 1)��Ҳ���������е�beta
    // �ڶ�������Ϊpose����ʾ����23���ؽڵ����λ�˼��ϸ��ǵķ���λ��(23+1=24, 3)��Ҳ���������е�theta
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
