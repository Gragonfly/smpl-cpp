#pragma once
#include <string>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
using Eigen::MatrixXi;
using Eigen::VectorXi;
using Eigen::VectorXd;
using Eigen::MatrixXd;

class Smpl
{
private:
	static const int face_index_num = 13776;
	static const int vertices_num = 6890;
	static const int joint_num = 23;
	static const int cal_joint_num = joint_num + 1;
	string model_path;
	string output_path;
	nlohmann::json model;
	MatrixXi face_indices;
	vector<MatrixXd> shapeblend;
	vector<MatrixXd> poseblend;
	MatrixXd vertices_template;
	MatrixXd joint_regressor;
	MatrixXi kinematic_tree;
	MatrixXd weights;
	MatrixXd poseBlendShape;
	MatrixXd shapeBlendShape;
	MatrixXd restShape;
	MatrixXd joints;
	vector<MatrixXd> poseRot_res;
	vector<MatrixXd> relativeTransformation;
	MatrixXd poseVertices_res;  

public:
	Smpl();
	Smpl(string& modelPath, string& outputPath);
	~Smpl();

	void loadModel();
	void launch(MatrixXd& shape, MatrixXd& pose);
	void blend(MatrixXd& shape, MatrixXd& pose);
	vector<MatrixXd> rodrigues(MatrixXd& pose);
	void regress();
	void transform();
	void skinning();
	void out();

};

