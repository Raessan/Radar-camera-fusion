#ifndef READ_MATRIX_FROM_FILE_HPP_
#define READ_MATRIX_FROM_FILE_HPP_

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <fstream>
#include <math.h>

// This function transforms a std::vector<std::vector<float>> to Eigen::MatrixXf
void vector2matrix(Eigen::MatrixXf &matrix, const std::vector<std::vector<float> > &vec)
{
    matrix.resize(vec.size(), vec[0].size());
	for (int i=0; i<(int)vec.size(); i++){
		for (int j=0; j<(int)vec[0].size(); j++){
			matrix(i,j) = vec[i][j];
		}
	}
}

// This function reads a file and fills a matrix. If the file can be read, returns 1, otherwise -1
int read_file(Eigen::MatrixXf &matrix, const std::string &file_path)
{
    std::vector<std::vector<float> > data;
	std::ifstream fileRead;
	std::string linea, tok;
	std::vector<float> line_data;

	fileRead.open(file_path.c_str());

	if (fileRead.is_open()){
		while (getline(fileRead, linea)) {
			std::stringstream iss(linea);
			line_data.clear();
			while (getline(iss, tok, ' ')) {
				line_data.push_back(strtod(tok.c_str(), NULL));
			}
			data.push_back(line_data);
		}
		fileRead.close();
		vector2matrix(matrix, data);
        return 1;
	}
	else
        //Could not open file
        return -1;
}

Eigen::Matrix<float, 3, Eigen::Dynamic> perspective_projection(const Eigen::Matrix<float, 3, Eigen::Dynamic> &pc_matrix, const Eigen::Matrix3f &projection){
	Eigen::Matrix<float, 3, Eigen::Dynamic> projected_matrix (3, pc_matrix.cols());
    for (int i=0; i<pc_matrix.cols(); i++){
		// Create the projected point
    	Eigen::Vector3f v_proj = projection*pc_matrix.col(i);
		// Normalize the result
		v_proj /= v_proj(2);
		// Recover original depth
		v_proj(2) = pc_matrix(2, i);
        projected_matrix.col(i) = v_proj;
    }
    return projected_matrix;
}


Eigen::Matrix<float, 3, Eigen::Dynamic> remove_outside(const Eigen::Matrix<float, 3, Eigen::Dynamic> &pc_matrix, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max){
	// Container for the filtered points
	std::vector<Eigen::Vector3f> filtered_points;

	// Loop through each point
	for (int i=0; i<pc_matrix.cols(); i++){
		float x = pc_matrix(0, i);
		float y = pc_matrix(1, i);
		float z = pc_matrix(2, i);

		// Check if the point is within the specified bounds
		if (x >= x_min && x <= x_max && y >= y_min && y <= y_max && z >= z_min && z <= z_max) {
            filtered_points.emplace_back(x, y, z);
        }

	}

	// Create the result matrix with the filtered points
    Eigen::MatrixXf filtered_pc(3, filtered_points.size());
    for (int i = 0; i < filtered_points.size(); ++i) {
        filtered_pc.col(i) = filtered_points[i];
    }

	return filtered_pc;
}


void get_error_lidar(const cv::Mat &depthmap, const Eigen::Matrix<float, 3, Eigen::Dynamic> &lidar, cv::Mat &depthmap_with_lidar, float &error, float radius = 1.0){
	// INITIALIZE VARIABLES
	// Error that will be calculated
	error = 0;
	// We initialize the depthmap with the lidar equal to the original depthmap
	depthmap_with_lidar = depthmap;
	// Get the number of points of the lidar
	int n_points = lidar.cols();
	// This will represent the X,Y,Z coordinates of each lidar point
	double x, y, z;
	
	// Loop over the points
	for (int i=0; i<n_points; i++){
		// Get the coordinates
        x = lidar(0,i);
        y = lidar(1,i);
        z = lidar(2,i);
		// Sum up the error
		error += std::abs(z - depthmap.at<float>(y,x));
		// Draw the circles
		cv::circle(depthmap_with_lidar, cv::Point(x, y), radius, cv::Scalar(z), cv::FILLED);
	}
	// Update error so that it's the mean error
	if (n_points > 0) error /= n_points;

	// Finally, we apply a colormap for the lidar depthmap
    cv::normalize(depthmap_with_lidar, depthmap_with_lidar, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(depthmap_with_lidar, depthmap_with_lidar, cv::COLORMAP_INFERNO);

}

#endif // READ_MATRIX_FROM_FILE_HPP_