#ifndef Classifier_H
#define Classifier_H

#include <fstream>
#include <utility>
#include <vector>
#include <iostream>

#include <cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/types_c.h>



class Classifier
{
public:
    struct TensorFlowDNNOutputStruct
    {
        std::string PlateValue;
        std::vector<int> ProbabilityVec;
    };
    Classifier();

    int init(std::string modelFilepath);
    TensorFlowDNNOutputStruct Forward(cv::Mat &img);


    void SetThreshold(float thresh);
    float GetThreshold();
private:

    float threshold = 0.5;
    Ort::Session *session;
    Ort::Env *env;
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    std::vector<Ort::Value> inputTensors;
    std::vector<std::vector<int64_t>> outputDims;
    std::vector<int64_t> inputDims;
};

#endif // Classifier_H

