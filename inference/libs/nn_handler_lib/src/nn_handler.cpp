#include "nn_handler_lib/nn_handler.hpp"

// Constructor with the filepath to the model .ONNX and the destination folder for the engine. The constructor reads the model and, if it exists, builds the engine in the destination folder.
NNHandler::NNHandler(std::string file_path_onnx, std::string file_path_destination, Precision p, int dla_core, int device_index){
    // Ensure the onnx model exists
    if (!Util::doesFileExist(file_path_onnx)) {
        std::cout << "Error: Unable to find file at path: " << file_path_onnx << std::endl;
        throw std::runtime_error("Error: Unable to find file!");
    }
    // Specify what precision to use for inference
    // FP16 is approximately twice as fast as FP32.
    options.precision = p;
    // If the model does not support dynamic batch size, then the below two parameters must be set to 1.
    // Specify the batch size to optimize for.
    options.optBatchSize = 1;
    // Specify the maximum batch size we plan on running.
    options.maxBatchSize = 1;
    // Specify the out file path
    options.out_file_path = file_path_destination;
    // Specify the DLA core (-1 does not activate it)
    options.dlaCore = dla_core;
    // Device index (number of GPU)
    options.deviceIndex = device_index;
    // Initialize the engine
    engine = std::make_shared<Engine>(options);
    

    // Build the onnx model into a TensorRT engine file.
    bool succ = engine->build(file_path_onnx);
    if (!succ) {
        throw std::runtime_error("Unable to build TRT engine.");
    }

    // Load the TensorRT engine file from disk
    succ = engine->loadNetwork();
    if (!succ) {
        throw std::runtime_error("Unable to load TRT engine.");
    }

    // Get features from the engine
    auto m_inputDims = engine->getInputDims();
    auto m_outputDims = engine->getOutputDims();
    auto m_IOTensorNames = engine->getTensorNames();

    // Fill input data
    n_inputs = m_inputDims.size();
    dims_in.resize(n_inputs);
    for (int i=0; i< n_inputs; i++){
        n_dim_in.push_back(m_inputDims[i].nbDims-1);
        int n_elems = 1;
        for (int j=1; j<m_inputDims[i].nbDims; j++){
            dims_in[i].push_back(m_inputDims[i].d[j]);
            n_elems *= m_inputDims[i].d[j];
        }
        n_elems_in.push_back(n_elems);
        input_names.push_back(m_IOTensorNames[i]);
    }

    // Fill output data
    n_outputs = m_outputDims.size();
    dims_out.resize(n_outputs);
    for (int i=0; i< n_outputs; i++){
        n_dim_out.push_back(m_outputDims[i].nbDims-1);
        int n_elems = 1;
        for (int j=1; j<m_outputDims[i].nbDims; j++){
            dims_out[i].push_back(m_outputDims[i].d[j]);
            n_elems *= m_outputDims[i].d[j];
        }
        n_elems_out.push_back(n_elems);
        output_names.push_back(m_IOTensorNames[n_inputs+i]);
    }

}


// Runs inference.
void NNHandler::run_inference(const std::vector<std::vector<float *>> &input, std::vector<std::vector<std::vector<float>>> &output, bool from_device){
    bool succ = engine->runInference(input, output, from_device);
    if (!succ) {
        throw std::runtime_error("Unable to run inference.");
    }
}
// Runs inference.
void NNHandler::run_inference(const std::vector<std::vector<float *>> &input, std::vector<std::vector<float>> &output, bool from_device){
    feature_vector.clear();
    run_inference(input, feature_vector);
    engine->transformOutput(feature_vector, output);
}
// Runs inference.
void NNHandler::run_inference(const std::vector<std::vector<float *>> &input, std::vector<float> &output, bool from_device){
    feature_vector.clear();
    run_inference(input, feature_vector);
    engine->transformOutput(feature_vector, output);
}
// Runs inference.
void NNHandler::run_inference(const std::vector<std::vector<float *>> &input, float *& output, bool from_device){
    feature_vector.clear();
    run_inference(input, feature_vector);
    output = std::move(&feature_vector[0][0][0]);
}
// Runs inference.
void NNHandler::run_inference(const float * input, std::vector<std::vector<std::vector<float>>> &output, bool from_device){
    assert(n_inputs==1 && options.maxBatchSize==1);
    auxiliar_input.clear();
    auxiliar_input.resize(1);
    auxiliar_input[0].resize(1);
    auxiliar_input[0][0] = const_cast<float*>(input);
    run_inference(auxiliar_input, output);
}
// Runs inference.
void NNHandler::run_inference(const float * input, std::vector<std::vector<float>> &output, bool from_device){
    assert(n_inputs==1 && options.maxBatchSize==1);
    auxiliar_input.clear();
    auxiliar_input.resize(1);
    auxiliar_input[0].resize(1);
    auxiliar_input[0][0] = const_cast<float*>(input);
    run_inference(auxiliar_input, output);
}
// Runs inference.
void NNHandler::run_inference(const float * input, std::vector<float> &output, bool from_device){
    assert(n_inputs==1 && options.maxBatchSize==1);
    auxiliar_input.clear();
    auxiliar_input.resize(1);
    auxiliar_input[0].resize(1);
    auxiliar_input[0][0] = const_cast<float*>(input);
    run_inference(auxiliar_input, output);
}
// Runs inference.
void NNHandler::run_inference(const float * input, float *& output, bool from_device){
    assert(n_inputs==1 && options.maxBatchSize==1);
    auxiliar_input.clear();
    auxiliar_input.resize(1);
    auxiliar_input[0].resize(1);
    auxiliar_input[0][0] = const_cast<float*>(input);
    run_inference(auxiliar_input, output);
}

// Prints the data of the handler related to inputs and outputs and their dimensions
void NNHandler::print_data(){
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "Num inputs: " << n_inputs << std::endl;
    for (int i=0; i< n_inputs; i++){
        std::cout << "Name input " << i+1 << ": " << input_names[i] << std::endl;
        std::cout << "Number of dimensions: " << n_dim_in[i] << std::endl;
        for (int j=0; j<n_dim_in[i]; j++){
            std::cout << "Dimension " << j+1 << ": " << dims_in[i][j] << std::endl;
        }
        std::cout << "----------------------------------------" << std::endl;
    }

    std::cout << "Num outputs: " << n_outputs << std::endl;
    for (int i=0; i< n_outputs; i++){
        std::cout << "Name output " << i+1 << ": " << output_names[i] << std::endl;
        std::cout << "Number of dimensions: " << n_dim_out[i] << std::endl;
        for (int j=0; j<n_dim_out[i]; j++){
            std::cout << "Dimension " << j+1 << ": " << dims_out[i][j] << std::endl;
        }
        std::cout << "----------------------------------------" << std::endl;
    }

}