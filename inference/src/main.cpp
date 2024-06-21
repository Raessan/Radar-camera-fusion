#include <iostream>
#include <eigen3/Eigen/Dense>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp> 
#include <set>
#include <random>
#include <memory>
#include <chrono>

#include "nn_handler_lib/nn_handler.hpp"
#include "inference/aux_functions.hpp"

// PARAMETERS

/** \brief Width of the original image  */
constexpr int w_original = 1600; 
/** \brief Height of the original image  */
constexpr int h_original = 900; 
/** \brief Width of the image that enters the depth anything NN */
constexpr int w_depth_anything = 518;
/** \brief Height of the image that enters the depth anything NN*/
constexpr int h_depth_anything = 518;
/** \brief Width of the image that enters the reldepth radar NN */
constexpr int w_nn_radarcam = 800;
/** \brief Height of the image that enters the reldepth radar NN */
constexpr int h_nn_radarcam = 450;
/** \brief Number of radar points to use. If the number of points in a sample is less than this, they are resampled
                randomly to create an array of fixed number of points. This shouldn't hurt the performance */
constexpr int nn_radar_points = 100;
/** \brief Threshold minimum depth to discard points */
constexpr double min_dist = 0.0;
/** \brief Threshold maximum depth to discard points */
constexpr double max_dist = 50.0;
/** \brief Number of inferences for warmup */
constexpr int n_inferences_warmup = 10;
/** \brief Number of inferences to calculate the average time */
constexpr int n_inferences = 10;
/** \brief Path of the depth anything model*/
const std::string path_model_depth_anything = "/ssd/Datasets_and_code/nuscenes_depth_estimation/code/models/depth_anything.onnx";
/** \brief Path of the model of the relative depth-radar algorithm */
const std::string path_model_radarcam = "/ssd/Datasets_and_code/nuscenes_depth_estimation/code/models/model_2024-06-20_15-22-48_epoch4.onnx";
/** \brief Path to save the TensorRT engine for inference*/
const std::string path_engine_save = "/ssd/Datasets_and_code/nuscenes_depth_estimation/code/models/";
/** \brief Path of the image to read*/
const std::string path_image = "/ssd/Datasets_and_code/nuscenes_depth_estimation/code/test_data/00100_im.jpg";
/** \brief Path of the radar pointcloud */
const std::string path_radar = "/ssd/Datasets_and_code/nuscenes_depth_estimation/code/test_data/00100_radar_pc.txt";
/** \brief Path of the lidar pointcloud */
const std::string path_lidar = "/ssd/Datasets_and_code/nuscenes_depth_estimation/code/test_data/00100_lidar_pc.txt";
/** \brief Path to camera matrix */
const std::string path_cam_matrix = "/ssd/Datasets_and_code/nuscenes_depth_estimation/code/test_data/00100_cam_matrix.txt";

// Other variables needed for inference
/** \brief Pointer of depth anything model */
std::unique_ptr<NNHandler> nn_handler_depth_anything;

/** \brief Pointer of reldepth_radar model */
std::unique_ptr<NNHandler> nn_handler_radarcam;

/** \brief Define random number generator */
std::random_device rd;
std::mt19937 mt;
std::uniform_int_distribution<int> distribution;


cv::Mat do_inference(cv::Mat image,const Eigen::Matrix<float, 3, Eigen::Dynamic> &radar){

    // Get number of points
    int n_points = radar.cols();
    
    // Only continue if the number of points is greater than 0
    if (n_points > 0){
        // radar_norm will contain the normalized radar 
        Eigen::MatrixXf radar_norm = radar; 
        // radar_adapted contains the final radar input, adapted to the size (3, nn_radar_points)
        // RowMajor layout is necessary to input the NN
        Eigen::Matrix<float, 3, Eigen::Dynamic, Eigen::RowMajor> radar_adapted;
        radar_adapted.resize(3, nn_radar_points);

        // Perform normalization
        for (int i=0; i<n_points; i++){
            radar_norm(0,i) /= w_nn_radarcam;
            radar_norm(1,i) /= h_nn_radarcam;
            radar_norm(2,i) = (radar_norm(2,i) - min_dist) / (max_dist - min_dist);
        }

        // This part adapts radar_norm to radar_adapted. If the number of points in a sample is less than nn_radar_points, the radar is resampled
        // randomly to create an array of fixed number of points. This shouldn't hurt the performance
        // If the number of points is greater, it selects a portion of the points
        if (n_points >= nn_radar_points) {
            // If the initial pointcloud has more points, select target_size points without replacement
            std::vector<int> indices(n_points);
            std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, ..., num_points-1

            std::shuffle(indices.begin(), indices.end(), mt); // Shuffle the indices

            // Select the first nn_radar_points indices
            indices.resize(nn_radar_points);

            // Extract selected indices from radar_adapted
            for (int i = 0; i < nn_radar_points; ++i) {
                radar_adapted.col(i) = radar_norm.col(indices[i]);
            }
        }
        else{
            // If the initial pointcloud has less points, randomly sample with replacement until filling target_size
            std::vector<int> selected_indices(nn_radar_points - n_points);
            distribution = std::uniform_int_distribution<int>(0, n_points - 1); // Define the distribution

            // Generate random indices
            for (int i = 0; i < nn_radar_points - n_points; ++i) {
                selected_indices[i] = distribution(mt);
            }

            // Concatenate the selected indices with existing indices
            radar_adapted.leftCols(radar_norm.cols()) = radar_norm;
            for (int i=0; i<selected_indices.size(); i++){
                radar_adapted.col(i+radar_norm.cols()) = radar_norm.col(selected_indices[i]);
            }
        }
        
        // Adapt data from image, so we have NCHW format, normalized and resized for the NN
        cv::Mat blob_im = cv::dnn::blobFromImage(image, 1.0/255.0, cv::Size(w_depth_anything, h_depth_anything), cv::Scalar(0.0, 0.0, 0.0), true, false, CV_32F);

        // We create the input of the DepthAnything NN, which requires further normalization to match the expected input
        float * input_nn_depth_anything = blob_im.ptr<float>();
        for (int i=0; i<w_depth_anything*h_depth_anything*3; i+=3){
            input_nn_depth_anything[i] = (input_nn_depth_anything[i] - 0.485f) / 0.229f;
            input_nn_depth_anything[i + 1] = (input_nn_depth_anything[i + 1] - 0.456f) / 0.224f;
            input_nn_depth_anything[i + 2] = (input_nn_depth_anything[i + 2] - 0.406f) / 0.225f;
        }

        // PERFORM INFERENCE WITH DEPTH ANYTHING!
        float * output_nn_depth_anything;
        nn_handler_depth_anything->run_inference(input_nn_depth_anything, output_nn_depth_anything);

        // Get the output in cv::Mat format
        cv::Mat im_output_depth_anything(h_depth_anything, w_depth_anything, CV_32FC1, output_nn_depth_anything);

        // Now we start preparing the input to the radar-cam NN. It requires a resize and normalization
        cv::Mat im_input_radarcam;
        // Resized image
        cv::resize(im_output_depth_anything, im_input_radarcam, cv::Size(w_nn_radarcam, h_nn_radarcam), cv::INTER_LINEAR);
        cv::normalize(im_input_radarcam, im_input_radarcam, 0, 1, cv::NORM_MINMAX, CV_32FC1);
        // This step is necessary because Depth Anything outputs inverses of depth
        im_input_radarcam = 1.0f - im_input_radarcam;

        // Input of 2nd NN
        std::vector<std::vector<float *>> input_radarcam(2);
        input_radarcam[0].resize(1);
        input_radarcam[1].resize(1);
        input_radarcam[0][0] = reinterpret_cast<float*>(im_input_radarcam.data);
        input_radarcam[1][0] = radar_adapted.data();

        // Now we create the output of the NN
        float * output_radarcam;

        // INFERENCE WITH THE RADAR_CAM NN!
        nn_handler_radarcam->run_inference(input_radarcam, output_radarcam);

        // We extract the cv::Mat from the pointer
        cv::Mat im_output_radarcam(h_nn_radarcam, w_nn_radarcam, CV_32FC1, output_radarcam);
        // Undo normalization so the output lies between min_dist and max_dist
        im_output_radarcam = im_output_radarcam * (max_dist - min_dist) + min_dist;

        return im_output_radarcam;
    }

    else{
        throw std::runtime_error("The pointcloud does not have any points!");
    }

}



int main(){

    // Matrix files
    Eigen::MatrixXf radar, lidar, cam_matrix;

    // Load image
    cv::Mat image = cv::imread(path_image);
    if (image.empty()){
        std::cerr << "Couldn't load image. Disconnecting..." << std::endl;
    }

    // Load matrix files
    if (read_file(radar, path_radar) == -1){
        std::cerr << "Couldn't load radar point cloud. Disconnecting..." << std::endl;
        return 1;
    }
    if (read_file(lidar, path_lidar) == -1){
        std::cerr << "Couldn't load lidar point cloud. Disconnecting..." << std::endl;
        return 1;
    }
    if (read_file(cam_matrix, path_cam_matrix) == -1){
        std::cerr << "Couldn't load camera matrix. Disconnecting..." << std::endl;
        return 1;
    }

    // Initialize inferencers
    nn_handler_depth_anything = std::make_unique<NNHandler>(path_model_depth_anything, path_engine_save, Precision::FP16);
    nn_handler_radarcam = std::make_unique<NNHandler>(path_model_radarcam, path_engine_save, Precision::FP16);

    // Transformation of the projection matrix after resizing image
    float scale_x = static_cast<float>(w_nn_radarcam)/w_original;
    float scale_y = static_cast<float>(h_nn_radarcam)/h_original;
    cam_matrix(0,0) *= scale_x;
    cam_matrix(1,1) *= scale_y;
    cam_matrix(0,2) *= scale_x;
    cam_matrix(1,2) *= scale_y;

    // Initialize random number generator
    mt.seed(rd());

    // Apply projection and filtering to both lidar and radar
    Eigen::MatrixXf radar_projected = perspective_projection(radar, cam_matrix);
    Eigen::MatrixXf lidar_projected = perspective_projection(lidar, cam_matrix);
    Eigen::MatrixXf radar_filtered = remove_outside(radar_projected, 1.0, w_nn_radarcam-1.0, 1.0, h_nn_radarcam-1.0, min_dist, max_dist);
    Eigen::MatrixXf lidar_filtered = remove_outside(lidar_projected, 1.0, w_nn_radarcam-1.0, 1.0, h_nn_radarcam-1.0, min_dist, max_dist);

    // Output of the inference
    cv::Mat output;

    // First warmup the NN
    for (int i=0; i<n_inferences_warmup; i++){
        output = do_inference(image, radar_filtered);
    }
    // Get the current time before inference
    auto start = std::chrono::high_resolution_clock::now();
    // And then perform inference as many times as requested
    for (int i=0; i<n_inferences; i++){
        output = do_inference(image, radar_filtered);
    }
    // Get the current time after inference
    auto end = std::chrono::high_resolution_clock::now();

     // Calculate the duration in milliseconds
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Time taken to perform inference: " << duration.count()/n_inferences << " milliseconds" << std::endl;

    // Now we plot it with lidar
    cv::Mat depthmap_with_lidar;
    float error_lidar;
    get_error_lidar(output, lidar_filtered, depthmap_with_lidar, error_lidar);

    std::cout << "Error lidar: " << error_lidar << std::endl;

    cv::imshow("Display Window", depthmap_with_lidar);
    cv::waitKey(0);
    cv::destroyWindow("Display Window");
    
    return 0;
}