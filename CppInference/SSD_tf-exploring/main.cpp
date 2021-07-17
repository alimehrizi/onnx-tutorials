// https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
// https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_cxx_api.h

#include<cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include <onnxruntime_session_options_config_keys.h>


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
#include "extra_utils.h"

#define CUDA_POST true

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

    std::cout << "Inference Execution Provider: CPU" << std::endl;


    std::string instanceName{"yolov5-inference"};
    std::string modelFilepath = "/home/altex/Mehrizi/konnect/tf_models/CarDetection-200x320.onnx";
    std::string imageFolderPath =   "/home/altex/test_images/images";
    std::string saveResultPath = "/home/altex/tfCar-result";
    std::string labelFilepath{"/home/altex/fake.txt"};;
    float score_threshold = 0.3;

    create_directory(saveResultPath);

    // https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L123
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                 instanceName.c_str());
    Ort::SessionOptions sessionOptions;





    bool enable_cpu_mem_arena = true;
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(sessionOptions, enable_cpu_mem_arena));
//    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);//controls whether the operators in the graph run sequentially or in parallel. Usually when a model has many branches, setting this option to false will provide better performance.
//    sessionOptions.SetIntraOpNumThreads(4); // number of threads used to parallelize the execution within nodes
//    sessionOptions.SetInterOpNumThreads(2); //number of threads used to parallelize the execution of the graph (across nodes).
    /*
     *When sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL,
     *  you can set sess_options.inter_op_num_threads to control the number of threads used to parallelize the execution of the graph (across nodes).
     */



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



    std::vector<const char*> inputNames{inputName};
    std::vector<const char*> outputNames;
    std::vector<Ort::Value> inputTensors;
    std::vector<std::vector<int64_t>> outputDims;


    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);



    for(int n=0;n<numOutputNodes;n++){
        const char* outputName = session.GetOutputName(n, allocator);
        std::cout << "Output Name: " << outputName << std::endl;
        outputNames.push_back(outputName);
        Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(n);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
        std::cout << "Output Type: " << outputType << std::endl;

        std::vector<int64_t> outputDim = outputTensorInfo.GetShape();
        std::cout << "Output Dimensions: " << outputDim << std::endl;
        outputDims.push_back(outputDim);

    }
    std::vector<std::vector<cv::Rect2f>> bboxes;
    std::vector<std::vector<float>> scores;
    std::vector<std::vector<int>> classes;


    std::vector<cv::Mat> images;
    std::vector<std::string> names;
    float t_p=0, t_all;
    Timer timer, timer2;
    int n_images=0;
    for (auto & entry : boost::filesystem::directory_iterator(imageFolderPath)){
        std::string img_path = entry.path().string();
        //if(n_images>300)break;
        auto img = cv::imread(img_path);
        if(img.empty())continue;
         names.push_back(img_path);
         images.push_back(img);
         std::cout<<img_path<<std::endl;
        timer2.Start();
        cv::Mat imageRGB;

        cv::cvtColor(img,imageRGB,cv::COLOR_BGR2RGB);

        std::vector<int64_t> inputDims = {1,imageRGB.size().height,imageRGB.size().width,3};
        auto inputTensorSize = vectorProduct(inputDims);
        inputTensors.clear();
        inputTensors.push_back(Ort::Value::CreateTensor<uint8_t>(
            memoryInfo, img.data, inputTensorSize, inputDims.data(),
            inputDims.size()));
         timer.Start();
        auto outputTensors = session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                            inputTensors.data(), 1, outputNames.data(), outputNames.size());
         t_p += timer.TimeSpent();

         std::vector<cv::Rect2f> bboxes_b;
         std::vector<float> scores_b;
         std::vector<int> classes_b;
        int num_detections = outputTensors[3].GetTensorData<float>()[0];
        for(int i=0;i<num_detections;i++){
            auto data = &outputTensors[0].GetTensorData<float>()[4*i];
            int cls = outputTensors[1].GetTensorData<float>()[i];
            float score = outputTensors[2].GetTensorData<float>()[i];
            if(score<score_threshold)
                continue;
            auto s = imageRGB.size();
            int xmin = data[1]*s.width;
            int xmax = data[3]*s.width;
            int ymin = data[0]*s.height;
            int ymax = data[2]*s.height;
            cv::Rect2d box(xmin,ymin,xmax-xmin,ymax-ymin);
            bboxes_b.push_back(cv::Rect2f(data[1],data[0],data[3]-data[1],data[2]-data[0]));
            classes_b.push_back(cls);
            scores_b.push_back(score);
            cv::rectangle(img,box,cv::Scalar(255),1);
        }
        cv::imwrite(saveResultPath+"/"+ std::to_string(n_images)+".jpg",img);
        t_all += timer2.TimeSpent();
        cv::imshow("result",img);
        cv::waitKey(0);

        bboxes.push_back(bboxes_b);
        scores.push_back(scores_b);
        classes.push_back(classes_b);
        n_images++;





    }
    std::cout<<"avg time = "<<t_all/n_images<<" ,avg graph time = "<<t_p/n_images<<std::endl;

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
