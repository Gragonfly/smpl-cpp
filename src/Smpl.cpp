#include "Smpl.h"

using namespace std;
using Eigen::MatrixXi;
using Eigen::VectorXi;
using Eigen::VectorXd;
using Eigen::MatrixXd;

Smpl::Smpl()
{
}

Smpl::Smpl(string& modelPath, string& outputPath)
{
	model_path = modelPath;
	output_path = outputPath;
}

Smpl::~Smpl()
{
}

void Smpl::regress() {
	cout << "regress()" << endl;
	// linearCombine
	cout << "vertices_template " << vertices_template.rows() << ", " << vertices_template.cols() << endl;
	cout << "poseBlendShape " << poseBlendShape.rows() << ", " << poseBlendShape.cols() << endl;
	cout << "shapeBlendShape " << shapeBlendShape.rows() << ", " << shapeBlendShape.cols() << endl;



	restShape = vertices_template + poseBlendShape + shapeBlendShape;
	cout << "restShape: " << restShape.rows() << " : " << restShape.cols() << endl;


	// joint regression
	MatrixXd jointShape = vertices_template + shapeBlendShape;  // 6890,3  joint_regressor(24, 6890)
	MatrixXd jointShape_t = jointShape.transpose();
	MatrixXd joint_regressor_t = joint_regressor.transpose();
	MatrixXd res = jointShape_t * joint_regressor_t;
	MatrixXd res_t = res.transpose();
	joints = res_t;

	cout << "joints : " << joints.rows() << " , " << joints.cols() << endl;


}

void Smpl::launch(MatrixXd& shape, MatrixXd& pose)
{
	// blend
	blend(shape, pose);

	// regress joints
	regress();
	

	// transform
	transform();



	// skinning
	skinning();


}

void Smpl::skinning() {

	// convert Homogeneous coordinates
	MatrixXd restShapeHomo(vertices_num, 4);
	restShapeHomo.block(0, 0, vertices_num, 3) = restShape;
	restShapeHomo.col(3) = Eigen::MatrixXd::Ones(vertices_num, 1);  //6890*4
	
	
	// LBS
	vector<MatrixXd> coefficients;
	MatrixXd lbs_tem(4, 4);
	for (int i = 0; i < vertices_num; i++) {
		VectorXd lbs_tem2 = weights.row(i);
		MatrixXd sum = Eigen::MatrixXd::Zero(4, 4);
		for (int j = 0; j < cal_joint_num; j++) {
			sum += lbs_tem2[j] * relativeTransformation[j];
		}
		coefficients.push_back(sum);
	}
	cout << "-----" << coefficients[2] << endl;


	// Cartesian coordinates 
	// vector<VectorXd>poseVertices
	vector<MatrixXd> poseVec;
	for (int i = 0; i < vertices_num; i++) {
		MatrixXd cart_tem(4, 1);
		cart_tem.col(0) = restShapeHomo.row(i).transpose();
		poseVec.push_back(coefficients[i] * cart_tem);
	}
	MatrixXd res(vertices_num, 3);
	double data1, data2, data3, data4;
	for (int i = 0; i < vertices_num; i++) {
		MatrixXd m = poseVec[i];
		data1 = m(0, 0);
		data2 = m(1, 0);
		data3 = m(2, 0);
		data4 = m(3, 0);
		data1 = data1 / data4;
		data2 = data2 / data4;
		data3 = data3 / data4;
		res.row(i) << data1, data2, data3;
	}
	poseVertices_res = res;
	// cout << "poseVertices" << res.row(2) << endl;

}

void Smpl::out()
{
	ofstream file(output_path);
	// face_indices (13776, 3)
	// poseVertices_res (6890, 3)

	//Eigen::Matrix<float, Eigen::Dynamic, 3> slice(poseVertices_res.data, vertices_num, 3);
	//Eigen::Matrix<int32_t, Eigen::Dynamic, 3> faceIndices(face_indices.data<int32_t>(), face_index_num, 3);

	for (int64_t i = 0; i < vertices_num; i++) {
		file << 'v' << ' '
			<< poseVertices_res(i, 0) << ' '
			<< poseVertices_res(i, 1) << ' '
			<< poseVertices_res(i, 2) << '\n';
	}

	for (int64_t i = 0; i < face_index_num; i++) {
		file << 'f' << ' '
			<< face_indices(i, 0) << ' '
			<< face_indices(i, 1) << ' '
			<< face_indices(i, 2) << '\n';
	}


}



void Smpl::transform() {

	// LocalTransform 

	vector<VectorXd> translation_temp;
	// cout << kinematic_tree.transpose() << endl;
	int ancestor = 0;
	translation_temp.push_back(joints.row(0)); // joints 24,3
	for (int i = 1; i < cal_joint_num; i++) {
		// find ancestor for node i
		for (int j = 0; j < cal_joint_num; j++) {
			if (kinematic_tree(j, 1) == i) {
				ancestor = kinematic_tree(j, 0);
				break;
			}
		}
		translation_temp.push_back(joints.row(i) - joints.row(ancestor));
	}
	//cout << "size : " << translation_temp.size() << endl;

	// Eigen库没有对应的扩充维度的方法，它只能是矩阵，所以下面要自己手动扩充，使得它成为齐次坐标
	vector<MatrixXd> translations;
	for (int i = 0; i < cal_joint_num; i++) {
		MatrixXd m = MatrixXd::Zero(4, 1);
		VectorXd tem(4);
		tem << translation_temp[i](0), translation_temp[i](1), translation_temp[i](2), 1;
		m.col(0) = tem;
		translations.push_back(m);
	}

	// cout << translations[2].size() << endl;  现在translations （24，4，1）

	// 下面实现postRotHomo的齐次坐标（24，4，3） 现在poseRot_res 是（24，3，3）
	vector<MatrixXd> postRotHomo;
	for (int i = 0; i < cal_joint_num; i++) {
		MatrixXd m(4, 3);
		m.block(0, 0, 3, 3) = poseRot_res[i];
		m.block(3, 0, 1, 3).setZero();
		postRotHomo.push_back(m);
	}
	// cout << postRotHomo[2] << endl;

	vector<MatrixXd> localTransformations;
	for (int i = 0; i < cal_joint_num; i++) {
		MatrixXd m(4, 4);
		m.block(0, 0, 4, 3) = postRotHomo[i];
		m.block(0, 3, 4, 1) = translations[i];
		localTransformations.push_back(m);
	}
	// cout << localTransformations[1] << endl;

	// Global Transform
	vector<MatrixXd> glotransformations;
	glotransformations.push_back(localTransformations[0]);
	ancestor = 0;
	MatrixXd glo_tem1;
	MatrixXd glo_tem2;
	MatrixXd glo_tem3;

	for (int i = 1; i < cal_joint_num; i++) {
		// find ancestor for node i
		for (int j = 0; j < cal_joint_num; j++) {
			if (kinematic_tree(j, 1) == i) {
				ancestor = kinematic_tree(j, 0);
				break;
			}
		}
		glo_tem1 = glotransformations[ancestor];
		glo_tem2 = localTransformations[i];
		glo_tem3 = glo_tem1 * glo_tem2;
		glotransformations.push_back(glo_tem3);
	}

	// cout << glotransformations[1] << endl;

	// relative Transform
	MatrixXd rel_tem1(3, 1);
	MatrixXd submatrix(3, 3);
	vector<MatrixXd> eliminated;
	for (int i = 0; i < cal_joint_num; i++) {
		submatrix = glotransformations[i].block<3, 3>(0, 0);
		rel_tem1 = joints.row(i).transpose();
		eliminated.push_back(submatrix * rel_tem1);
	}

	vector<MatrixXd> eliminatedHomo;
	MatrixXd e_tem(4, 4);
 	for (int i = 0; i < cal_joint_num; i++) {
		e_tem.setZero();
		e_tem.topRightCorner<3, 1>() = eliminated[i];
		eliminatedHomo.push_back(e_tem);
	}

	vector<MatrixXd> relativeTransformations;
	for (int i = 0; i < cal_joint_num; i++) {
		relativeTransformations.push_back(glotransformations[i] - eliminatedHomo[i]);
	}

	// cout << relativeTransformations[1] << endl;


	relativeTransformation = relativeTransformations;


}



void Smpl::blend(MatrixXd& shape, MatrixXd& pose)
{
	// Pose blend 24*3
	//cout << "dhwiadha" << pose << endl;
	vector<MatrixXd> poseRot = rodrigues(pose);
	//cout << "----------------" << poseRot[2] << endl;
	poseRot_res = poseRot;

	// 计算T-pose下的原始旋转矩阵
	vector<MatrixXd> restposeRot;
	for (int i = 0; i < cal_joint_num; i++) {
		restposeRot.push_back(Eigen::Matrix3d::Identity());
	}

	//  pose blend coefficients
	VectorXd unposeRot(cal_joint_num * 9);
	int startIndex = 0;
	int numElementsMatrix = 0;
	for (int i = 0; i < cal_joint_num; i++) {
		numElementsMatrix = poseRot[i].size();
		VectorXd matrixVec = Eigen::Map<VectorXd>(poseRot[i].data(), numElementsMatrix);
		unposeRot.segment(startIndex, numElementsMatrix) = matrixVec;
		startIndex += numElementsMatrix;
	}


	startIndex = 0;
	VectorXd unrestposeRot(cal_joint_num * 9);
	for (int i = 0; i < restposeRot.size(); i++) {
		numElementsMatrix = restposeRot[i].size();
		VectorXd matrixVec = Eigen::Map<VectorXd>(restposeRot[i].data(), numElementsMatrix);
		unrestposeRot.segment(startIndex, numElementsMatrix) = matrixVec;
		startIndex += numElementsMatrix;
	}


	unposeRot = unposeRot.tail(cal_joint_num * 9 - 9);
	unrestposeRot = unrestposeRot.tail(cal_joint_num * 9 - 9);

	VectorXd poseBlendCoeffs = unposeRot - unrestposeRot;

	//cout << poseBlendCoeffs << endl;

	// pose blend
	MatrixXd _poseBlendShape(vertices_num, 3);
	VectorXd temp(joint_num * 9);
	double res = 0;
	for (int i = 0; i < vertices_num; i++) {
		for (int j = 0; j < 3; j++) {
			temp = poseblend[i].row(j);
			res = temp.dot(poseBlendCoeffs);
			_poseBlendShape(i, j) = res;
		}
	}
	poseBlendShape = _poseBlendShape;
	//至此论文中的Bp（theta）已经算好了

	// Shape blend 

	MatrixXd _shapeblend(vertices_num, 3);
	VectorXd temp1(shape.rows());
	temp1 = shape.col(0);
	VectorXd _temp(10);
	for (int i = 0; i < vertices_num; i++) {
		for (int j = 0; j < 3; j++) {
			_temp = shapeblend[i].row(j);
			res = _temp.dot(temp1);
			_shapeblend(i, j) = res;
		}
	}
	shapeBlendShape = _shapeblend;
	
	//至此论文中的Bs（beta）已经算好了

}


vector<MatrixXd> Smpl::rodrigues(MatrixXd& pose)
{
	vector<MatrixXd> rotation;
	const double eps = 1e-8;
	Eigen::Vector3d theta;
	Eigen::Vector3d axis;
	MatrixXd rotation_mat;
	MatrixXd Rk(3, 3);		//用来存储矩阵形式的旋转向量
	for (int i = 0; i < cal_joint_num; i++) {
		theta = pose.row(i);
		double angle = theta.norm();	//计算旋转角度
		axis = theta / (angle + eps);	//计算旋转轴
		Rk << 0, -axis[2], axis[1], axis[2], 0, -axis[0], -axis[1], axis[0], 0; //计算Rk
		rotation_mat = Eigen::Matrix3d::Identity() + Rk * sin(angle) + Rk * Rk * (1 - cos(angle)); //计算旋转矩阵
		rotation.push_back(rotation_mat);
		//cout <<"i==" << i <<  rotation_mat << endl;
	}
	return rotation;

}


void Smpl::loadModel()
{
	ifstream file(model_path);
	file >> model;

	/*
	*  我们先来看一下json中都保存了哪些模型参数：
	*  vertices_template：标准模型的全局顶点位置 - (6890, 3) 应该是T-pose姿态下的骨骼坐标（以骨骼根节点为原点的坐标系）
	*  face_indices: 每个面（三角形）的顶点index - (13776, 3)
    *  joint_regressor: 联合回归器 - (24, 6890)
    *  kinematic_tree_table: 运动树的表格 - (2, 24)		
    *  weights: 用于线性混合皮肤的权重 - (6890, 24)		LBS的蒙皮权重，W_{k, i}
    *  shape_blend_shapes: 形状混合形状 - (6890, 3, 10)  每种体态对应的顶点骨骼坐标
    *  pose_blend_shapes: 姿势混合形状 - (6890, 3, 207)  每种位姿下的顶点骨骼坐标
	* 
	*/

	// vertices_template (6890, 3)
	MatrixXd vertices_template_tem(vertices_num, 3);
	vector<vector<double>> vertices_template_vec = model["vertices_template"];
	for (int i = 0; i < vertices_num; i++) {
		vertices_template_tem.row(i) = VectorXd::Map(&vertices_template_vec[i][0], vertices_template_vec[i].size());
	}
	vertices_template = vertices_template_tem;


	// face_indices (13776, 3)
	MatrixXi face_indices_tem(face_index_num, 3);
	vector<vector<int>> faceIndices = model["face_indices"];
	for (int i = 0; i < face_index_num; i++) {
		face_indices_tem.row(i) = VectorXi::Map(&faceIndices[i][0], faceIndices[i].size());
	}
	face_indices = face_indices_tem;


	// pose (6890, 3, 207) && shape (6890, 3, 10)
	vector<vector<vector<double>>> shape_blend = model["shape_blend_shapes"];
	vector<double> temp;
	vector<vector<double>> temp2;
	MatrixXd shape_tem(3, 10);
	
	for (int i = 0; i < vertices_num; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 10; k++) {
				temp.push_back(shape_blend[i][j][k]);
			}
			temp2.push_back(temp);
			temp.clear();
		}
		for (int q = 0; q < 3; q++) {
			shape_tem.row(q) = VectorXd::Map(&temp2[q][0], temp2[q].size());
		}
		temp2.clear();
		shapeblend.push_back(shape_tem);
	}

	vector<vector<vector<double>>> pose_blend = model["pose_blend_shapes"];
	MatrixXd pose_tem(3, joint_num * 9);

	for (int i = 0; i < vertices_num; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < joint_num * 9; k++) {
				temp.push_back(pose_blend[i][j][k]);
			}
			temp2.push_back(temp);
			temp.clear();
		}
		for (int q = 0; q < 3; q++) {
			pose_tem.row(q) = VectorXd::Map(&temp2[q][0], temp2[q].size());
		}
		temp2.clear();
		poseblend.push_back(pose_tem);
	}
	
	// regressor  (24, 6890)
	MatrixXd regressor_tem(cal_joint_num, vertices_num);
	vector<vector<double>> regressor_vec = model["joint_regressor"];
	for (int i = 0; i < cal_joint_num; i++) {
		regressor_tem.row(i) = VectorXd::Map(&regressor_vec[i][0], regressor_vec[i].size());
	}
	joint_regressor = regressor_tem;

	// kinematic_tree  (2, 24)
	MatrixXi kinematic_tree_tem(2, cal_joint_num);
	vector<vector<int>> kinematic_tree_vec = model["kinematic_tree"];
	for (int i = 0; i < 2; i++) {
		kinematic_tree_tem.row(i) = VectorXi::Map(&kinematic_tree_vec[i][0], kinematic_tree_vec[i].size());
	}
	kinematic_tree = kinematic_tree_tem.transpose();

	// skinner  (6890, 24)
	MatrixXd weights_tem(vertices_num, cal_joint_num);
	vector<vector<double>> weights_vec = model["weights"];
	for (int i = 0; i < vertices_num; i++) {
		weights_tem.row(i) = VectorXd::Map(&weights_vec[i][0], weights_vec[i].size());
	}
	weights = weights_tem;
	

}


