// https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
// https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_cxx_api.h
#include <cuda_provider_factory.h>
#include<cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>


#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <boost/filesystem.hpp>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include"cuda_utils.h"
#include "extra_utils.h"

#define CUDA_POST false

std::vector<std::string> split(std::string s,std::string delimiter = ">="){

    std::vector<std::string> results;

    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        results.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    results.push_back(s);
    return results;
}

std::vector<float> preprocess_img(cv::Mat& src, cv::Mat &dst, float out_w, float out_h) {
    float in_h = src.size().height;
    float in_w = src.size().width;
    std::vector<float> pad_info;
    float scale = std::min(out_w / in_w, out_h / in_h);
    pad_info.push_back(scale);

    int mid_h = static_cast<int>(in_h * scale);
    int mid_w = static_cast<int>(in_w * scale);

    cv::resize(src, dst, cv::Size(mid_w, mid_h));

    int top = (static_cast<int>(out_h) - mid_h) / 2;
    int down = (static_cast<int>(out_h)- mid_h + 1) / 2;
    int left = (static_cast<int>(out_w)- mid_w) / 2;
    int right = (static_cast<int>(out_w)- mid_w + 1) / 2;
    pad_info.push_back(left);
    pad_info.push_back(top);
    cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    pad_info.push_back(in_w);
    pad_info.push_back(in_h);
    return pad_info;
}

int vectorArgMax(std::vector<float> data){
    int max_id=0;
    float max_value=-INFINITY;
    for(int i=0;i<data.size();i++){
        if(data[i]>max_value){
            max_value = data[i];
            max_id = i;
        }
    }
    return max_id;
}

void postProcessCpu(float *result,std::vector<int64_t> outputDims, std::vector<std::vector<cv::Rect2f>> &final_bboxes, std::vector<std::vector<float>> &final_scores, std::vector<std::vector<int>> &final_classes, float conf_thresh, float iou_thresh, std::vector<float> pad_info){

    for(int b=0;b<outputDims[0];b++){
        std::vector<float> scores;
        std::vector<int> classes;
        std::vector<cv::Rect> bboxes;
        std::vector<cv::Rect> bboxes_offsets;
        int num_classes = outputDims[2] - 5;
        int offset = b*outputDims[1]*outputDims[2];
        for(int i=offset;i<offset+outputDims[1]*outputDims[2];i+=outputDims[2]){
            float conf = result[i+4];
            if(conf<conf_thresh)
                continue;
            cv::Rect2f box;
            box.x = result[i] - result[i+2]/2;
            box.y = result[i+1] - result[i+3]/2;
            box.width = result[i+2];
            box.height = result[i+3];
            std::vector<float> temp(num_classes);
            temp.assign(&result[i+5],&result[i+outputDims[2]]);
            int cls = vectorArgMax(temp);
            classes.push_back(cls);
            scores.push_back(conf*temp[cls]);
            bboxes.push_back(box);
            auto offset_box = box;
            offset_box.x += cls * 2000;
            bboxes_offsets.push_back(offset_box);

        }
        final_bboxes[b].clear();
        final_classes[b].clear();
        final_scores[b].clear();
        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(bboxes_offsets, scores, conf_thresh, iou_thresh, nms_indices);
        for(int idx:nms_indices){
            cv::Rect2f box = bboxes[idx];
            box.y -= pad_info[2];
            box.x -= pad_info[1];
            box.x /= pad_info[0];
            box.y /= pad_info[0];
            box.width /= pad_info[0];
            box.height /= pad_info[0];
            box.y = (box.y + box.height/2)/pad_info[4];
            box.x = (box.x + box.width/2)/pad_info[3];
            box.height /= pad_info[4];
            box.width /= pad_info[3];
            final_scores[b].push_back(scores[idx]);
            final_bboxes[b].push_back(box);
            final_classes[b].push_back(classes[idx]);
        }
    }

}

void postProcessCuda(float *result,std::vector<int64_t> outputDims, std::vector<std::vector<cv::Rect2f>> &final_bboxes, std::vector<std::vector<float>> &final_scores, std::vector<std::vector<int>> &final_classes, float conf_thresh, float iou_thresh, std::vector<float> pad_info){
    final_bboxes.clear();
    final_classes.clear();
    final_scores.clear();
    float *nms_result = (float*)malloc(outputDims[0]*NUM_DET_ATTR*MAX_DET_NMS*sizeof(float));
    cudaNMS(result,outputDims[0],outputDims[1],outputDims[2],nms_result);
    for(int b=0;b<outputDims[0];b++){
        final_bboxes[b].clear();
        final_classes[b].clear();
        final_scores[b].clear();
        int offset = b*MAX_DET_NMS*NUM_DET_ATTR;
        for(int i=offset;i<offset+MAX_DET_NMS*NUM_DET_ATTR;i+=NUM_DET_ATTR){
            float conf = nms_result[i+4];
            if(conf<0.3)
                continue;
            cv::Rect2f box;
            box = cv::Rect2f(cv::Point(nms_result[i+0], nms_result[i+1]),
                                cv::Point(nms_result[i+2], nms_result[i+3]));
            box.y -= pad_info[2];
            box.x -= pad_info[1];
            box.x /= pad_info[0];
            box.y /= pad_info[0];
            box.width /= pad_info[0];
            box.height /= pad_info[0];
            box.y = (box.y + box.height/2)/pad_info[4];
            box.x = (box.x + box.width/2)/pad_info[3];
            box.height /= pad_info[4];
            box.width /= pad_info[3];
            final_scores[b].push_back(conf);
            final_classes[b].push_back(nms_result[i+5]);
            final_bboxes[b].push_back(box);
            //std::cout<<"conf="<<conf<<" ,class="<<nms_result[i+5]<<" ,box="<<box<<std::endl;
        }
    }
}

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

/**
 * @brief Print ONNX tensor data type
 * https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L93
 * @param os
 * @param type
 * @return std::ostream&
 */
std::ostream& operator<<(std::ostream& os,
                         const ONNXTensorElementDataType& type)
{
    switch (type)
    {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            os << "undefined";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            os << "float";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            os << "uint8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            os << "int8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            os << "uint16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            os << "int16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            os << "int32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            os << "int64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            os << "std::string";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            os << "bool";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            os << "float16";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            os << "double";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            os << "uint32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            os << "uint64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            os << "float real + float imaginary";
            break;
        case ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            os << "double real + float imaginary";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            os << "bfloat16";
            break;
        default:
            break;
    }

    return os;
}

std::vector<std::string> readLabels(std::string& labelFilepath)
{
    std::vector<std::string> labels;
    std::string line;
    std::ifstream fp(labelFilepath);
    while (std::getline(fp, line))
    {
        labels.push_back(line);
    }
    return labels;
}

int main(int argc, char* argv[])
{
    bool useCUDA{true};
    const char* useCUDAFlag = "--use_cuda";
    const char* useCPUFlag = "--use_cpu";
    if (argc == 1)
    {
        useCUDA = false;
    }
    else if ((argc == 2) && (strcmp(argv[1], useCUDAFlag) == 0))
    {
        useCUDA = true;
    }
    else if ((argc == 2) && (strcmp(argv[1], useCPUFlag) == 0))
    {
        useCUDA = false;
    }
    else if ((argc == 2) && (strcmp(argv[1], useCUDAFlag) != 0))
    {
        useCUDA = false;
    }
    else
    {
        throw std::runtime_error{"Too many arguments."};
    }

    if (useCUDA)
    {
        std::cout << "Inference Execution Provider: CUDA" << std::endl;
    }
    else
    {
        std::cout << "Inference Execution Provider: CPU" << std::endl;
    }

    std::string instanceName{"yolov5-inference"};
    std::string modelFilepath{"/home/altex/Mehrizi/Models/TODv1.0-320s/exp/weights/best.onnx"};
    std::string imageFolderPath =  "/home/altex/test_images/images";
    std::string labelFilepath{"/home/altex/fake.txt"};
    std::string saveResultPath = "/home/altex/test_images/onnx-result";

    std::vector<std::string> labels{readLabels(labelFilepath)};

    // https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L123
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                 instanceName.c_str());
    Ort::SessionOptions sessionOptions;




    if (useCUDA)
    {
        // Using CUDA backend
        // https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/providers/cuda/cuda_provider_factory.h#L13
        OrtStatus* status =
            OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
    }else
    {
        bool enable_cpu_mem_arena = true;
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(sessionOptions, enable_cpu_mem_arena));
        sessionOptions.SetIntraOpNumThreads(8); // controls the number of threads to use to run the model
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);//controls whether the operators in the graph run sequentially or in parallel. Usually when a model has many branches, setting this option to false will provide better performance.
        /*
         *When sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL,
         *  you can set sess_options.inter_op_num_threads to control the number of threads used to parallelize the execution of the graph (across nodes).
         */

    }

    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node
    // removals) ORT_ENABLE_EXTENDED -> To enable extended optimizations
    // (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible optimizations
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, modelFilepath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();

    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;

    const char* inputName = session.GetInputName(0, allocator);
    std::cout << "Input Name: " << inputName << std::endl;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    std::cout << "Input Type: " << inputType << std::endl;

    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    std::cout << "Input Dimensions: " << inputDims << std::endl;

    for(int i=0;i<numOutputNodes;i++){
        const char* outputName = session.GetOutputName(i, allocator);
        std::cout << "Output Name: " << outputName << std::endl;

        Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(i);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
        std::cout << "Output Type: " << outputType << std::endl;

        std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
        std::cout << "Output Dimensions: " << outputDims << std::endl;
    }

    const char* outputName = session.GetOutputName(0, allocator);
    std::cout << "Output Name: " << outputName << std::endl;

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    std::cout << "Output Type: " << outputType << std::endl;

    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    std::cout << "Output Dimensions: " << outputDims << std::endl;
    size_t inputTensorSize = vectorProduct(inputDims);
    size_t outputTensorSize = vectorProduct(outputDims);
//    assert(("Output tensor size should equal to the label set size.",
//            labels.size() == outputTensorSize));
    //std::vector<float> outputTensorValues(outputTensorSize);

    float *outputTensorValues;
    if(useCUDA & CUDA_POST){
        HANDLE_ERROR(cudaMalloc((void**)&outputTensorValues,int(outputTensorSize)*sizeof(float)));
    }else{
        outputTensorValues = (float*)malloc(outputTensorSize*sizeof(float));

    }
    std::vector<const char*> inputNames{inputName};
    std::vector<const char*> outputNames{outputName};


    float *input_data = (float*)malloc(vectorProduct(inputDims)*sizeof(float));

    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, input_data, inputTensorSize, inputDims.data(),
        inputDims.size()));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, outputTensorValues, outputTensorSize,
        outputDims.data(), outputDims.size()));

    std::vector<std::vector<cv::Rect2f>> bboxes;
    std::vector<std::vector<float>> scores;
    std::vector<std::vector<int>> classes;


    std::vector<cv::Mat> images;
    std::vector<std::string> names;
    std::vector<float> pad_info;
    float t_p=0;
    Timer timer;
    int n_images=0;
    for (auto & entry : boost::filesystem::directory_iterator(imageFolderPath)){
        std::string img_path = entry.path().string();
        //if(n_images>300)break;
        auto img = cv::imread(img_path);
        if(img.empty())continue;
         names.push_back(img_path);
         images.push_back(img);
         std::cout<<img_path<<std::endl;

        cv::Mat resizedImageBGR;
        cv::Mat resizedImage;
        cv::Mat preprocessedImage;
        timer.Start();
        pad_info = preprocess_img(img,resizedImageBGR, inputDims.at(2), inputDims.at(3) );
        resizedImageBGR.convertTo(resizedImage, CV_32F, 1.0 / 255.0);

        // HWC to CHW
        cv::dnn::blobFromImage(resizedImage, preprocessedImage);

        memcpy(input_data, preprocessedImage.data,vectorProduct(inputDims)*sizeof(float));
        // https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_cxx_api.h#L353

        cudaDeviceSynchronize();
        session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                    inputTensors.data(), 1, outputNames.data(),
                    outputTensors.data(), 1);

        std::vector<std::vector<cv::Rect2f>> final_bboxes(outputDims[0]);
        std::vector<std::vector<float>> final_scores(outputDims[0]);
        std::vector<std::vector<int>> final_classes(outputDims[0]);
        if(useCUDA & CUDA_POST){
            postProcessCuda(outputTensorValues,outputDims,final_bboxes, final_scores,final_classes,CONF_THR,IOU_THR, pad_info);
        }else{
            postProcessCpu(outputTensorValues,outputDims,final_bboxes, final_scores,final_classes,CONF_THR,IOU_THR, pad_info);
        }
        cudaDeviceSynchronize();
        t_p += timer.TimeSpent();
        bboxes.push_back(final_bboxes[0]);
        scores.push_back(final_scores[0]);
        classes.push_back(final_classes[0]);
        n_images++;
        for(int i=0;i<final_bboxes[0].size();i++){
           auto box = final_bboxes[0][i];
           box.x = (box.x - box.width/2)*img.size().width;
           box.y = (box.y - box.height/2)*img.size().height;
           box.width *= img.size().width;
           box.height *= img.size().height;
           cv::rectangle(img, box,cv::Scalar(255),2);
        }
        cv::imshow("result", img);
        cv::waitKey(0);





    }


    std::cout<<"graph average running time = "<<t_p/names.size()<< "ms "<<std::endl;

    for(int b=0;b<bboxes.size();b++){
        std::ofstream result_file;
        std::string image_name = split(split(names[b],"/").back(),".")[0];
        std::string result_path = saveResultPath + "/"+image_name+".txt";
        //std::cout<<result_path<<std::endl;
        result_file.open(result_path,std::ios::out);

        for(int i=0;i<bboxes[b].size();i++){
            auto box = bboxes[b][i];
            result_file<<box.x<<" "<<box.y<<" "<<box.width<<" "<<box.height<<" "<<scores[b][i]<<" "<<classes[b][i]<<"\n";
        }
        result_file.close();
    }

//    for(int b=0;b<images.size();b++){
//        auto image_size = images[b].size();
//        for(int i=0;i<bboxes[b].size();i++){
//            auto box = bboxes[b][i];
//            box.x = (box.x - box.width/2)*image_size.width;
//            box.y = (box.y - box.height/2)*image_size.height;
//            box.width *= image_size.width;
//            box.height *= image_size.height;
//            cv::rectangle(images[b], box,cv::Scalar(255),2);
//        }
//    }
//    for(int b=0;b<images.size() && b<4;b++)
//        cv::imshow("frame-"+std::to_string(b),images[b]);
//    cv::waitKey(0);


}
