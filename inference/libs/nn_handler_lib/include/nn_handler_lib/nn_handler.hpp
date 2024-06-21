#ifndef NN_HANDLER_HPP_
#define NN_HANDLER_HPP_

#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <cstring>

#include "tensorrt_cpp_api/engine.h"

/** 
 * @brief Class that creates a NN handler with TensorRT to build and run inference with an ONNX model, and to keep information about inputs and outputs. This class assumes that the batch size is always 1
 * 
 */
class NNHandler{
    
    public:
        /**
         * @brief Constructor with the filepath to the model .ONNX and the destination folder for the engine. The constructor reads the model and, if it exists, builds the engine in the destination folder.
         * @param file_path_onnx Path to the onnx file
         * @param file_path_destination Path to the destination file
         * @param p Precision of the NN. Default: FP16
         * @param dla_core DLA core to use. By default, it does not use it (by setting it to -1)
         * @param device_index GPU index (ORIN only has the 0 index)
        */
        NNHandler(std::string file_path_onnx, std::string file_path_destination, Precision p=Precision::FP16, int dla_core = -1, int device_index=0);
        /**
         * @brief Runs inference.
         * @param input Double vector of float pointer with the input of the NN. This is the standard input of the engine
         * @param output Triple vector of float with the output of the NN. This is the standard output of the engine
         * @param from_device If this is true, the input pointer is expected in the GPU
        */
        void run_inference(const std::vector<std::vector<float *>> &input, std::vector<std::vector<std::vector<float>>> &output, bool from_device=false);
        /**
         * @brief Runs inference.
         * @param input Double vector of float pointer with the input of the NN. This is the standard input of the engine
         * @param output Double vector of float with the output of the NN. THIS IS ONLY POSSIBLE IF BATCH_SIZE=1
         * @param from_device If this is true, the input pointer is expected in the GPU
        */
        void run_inference(const std::vector<std::vector<float *>> &input, std::vector<std::vector<float>> &output, bool from_device=false);
        /**
         * @brief Runs inference.
         * @param input Double vector of float pointer with the input of the NN. This is the standard input of the engine
         * @param output vector of float with the output of the NN. THIS IS ONLY POSSIBLE IF num_outputs=1 AND batch_size=1
         * @param from_device If this is true, the input pointer is expected in the GPU
        */
        void run_inference(const std::vector<std::vector<float *>> &input, std::vector<float> &output, bool from_device=false);
        /**
         * @brief Runs inference.
         * @param input Double vector of float pointer with the input of the NN. This is the standard input of the engine
         * @param output Float pointer with the output of the NN. THIS IS ONLY POSSIBLE IF num_outputs=1 AND batch_size=1.
         * @param from_device If this is true, the input pointer is expected in the GPU
        */
        void run_inference(const std::vector<std::vector<float *>> &input, float *& output, bool from_device=false);
        /**
         * @brief Runs inference.
         * @param input Float pointer with the input of the NN. THIS IS ONLY POSSIBLE IF N_INPUTS=1. IMPLICITLY, IT IS CONSIDERED BATCH_SIZE=1
         * @param output Triple vector of float with the output of the NN. This is the standard output of the engine
         * @param from_device If this is true, the input pointer is expected in the GPU
        */
        void run_inference(const float * input, std::vector<std::vector<std::vector<float>>> &output, bool from_device=false);
        /**
         * @brief Runs inference.
         * @param input Float pointer with the input of the NN. THIS IS ONLY POSSIBLE IF N_INPUTS=1. IMPLICITLY, IT IS CONSIDERED BATCH_SIZE=1
         * @param output Double vector of float with the output of the NN. THIS IS ONLY POSSIBLE IF BATCH_SIZE=1
         * @param from_device If this is true, the input pointer is expected in the GPU
        */
        void run_inference(const float * input, std::vector<std::vector<float>> &output, bool from_device=false);
        /**
         * @brief Runs inference.
         * @param input Float pointer with the input of the NN. THIS IS ONLY POSSIBLE IF N_INPUTS=1. IMPLICITLY, IT IS CONSIDERED BATCH_SIZE=1
         * @param output vector of float with the output of the NN. THIS IS ONLY POSSIBLE IF num_outputs=1 AND batch_size=1
         * @param from_device If this is true, the input pointer is expected in the GPU
        */
        void run_inference(const float * input, std::vector<float> &output, bool from_device=false);
        /**
         * @brief Runs inference.
         * @param input Float pointer with the input of the NN. THIS IS ONLY POSSIBLE IF N_INPUTS=1. IMPLICITLY, IT IS CONSIDERED BATCH_SIZE=1
         * @param output Float pointer with the output of the NN. THIS IS ONLY POSSIBLE IF num_outputs=1 AND batch_size=1.
         * @param from_device If this is true, the input pointer is expected in the GPU
        */
        void run_inference(const float * input, float *& output, bool from_device=false);
        

        /**
         * @brief Returns the number of inputs to the NN
         * @return n_inputs
        */
        inline int get_n_inputs(){return n_inputs;}
        /**
         * @brief Returns the number of outputs to the NN
         * @return n_outputs
        */
        inline int get_n_outputs(){return n_outputs;}
        /**
         * @brief Returns the sizes of the inputs of the NN
         * @return n_dim_in
        */
        inline std::vector<std::vector<int>> get_dimensions_in(){return dims_in;}
        /**
         * @brief Returns the sizes of the outputs of the NN
         * @return n_dim_out
        */
        inline std::vector<std::vector<int>> get_dimensions_out(){return dims_out;}
        /**
         * @brief Returns the number of elems of the input to the NN
         * @return n_elems_in
        */
        inline std::vector<int> get_n_elems_in(){return n_elems_in;}
        /**
         * @brief Returns the number of elems of the output to the NN
         * @return n_elems_out
        */
        inline std::vector<int> get_n_elems_out(){return n_elems_out;}
        /**
         * @brief Returns the numbers of the inputs
         * @return input_names
        */
        inline std::vector<std::string> get_input_names(){return input_names;}
        /**
         * @brief Returns the numbers of the outputs
         * @return output_names
        */
        inline std::vector<std::string> get_output_names(){return output_names;}
        /**
         * @brief Returns the batch_size
         * @return batch_size
        */
        inline int get_batch_size(){return batch_size;}

        /**
         * @brief Prints the data of the handler related to inputs and outputs and their dimensions
        */
        void print_data();

        
    private:
        /** \brief Number of inputs */
        int n_inputs;
        /** \brief Sizes  of each input */
        std::vector<int> n_dim_in, n_elems_in;
        /** \brief Dimension sizes of each input */
        std::vector<std::vector<int>> dims_in;
        /** \brief Names of each input */
        std::vector<std::string> input_names;
        /** \brief Number of outputs */
        int n_outputs;
        /** \brief Sizes of each output */
        std::vector<int> n_dim_out, n_elems_out;
        /** \brief Dimension sizes of each output */
        std::vector<std::vector<int>> dims_out;
        /** \brief Names of each output */
        std::vector<std::string> output_names;
        /** \brief Batch size */
        int batch_size;

        /** \brief Options of the NN */
        Options options;
        

        /** \brief Auxiliar feature vector */
        std::vector<std::vector<std::vector<float>>> feature_vector;
        /** \brief Auxiliar input vector */
        std::vector<std::vector<float *>> auxiliar_input;
        /** \brief Engine of the NN */
        std::shared_ptr<Engine> engine;

};

#endif // NN_HANDLER_HPP_