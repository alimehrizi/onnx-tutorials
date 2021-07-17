#ifndef Detector_H
#define Detector_H

#include <fstream>
#include <utility>
#include <vector>
#include <iostream>

#include<cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include <onnxruntime_session_options_config_keys.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


struct Detection
{
    float time;
    cv::Rect2f bbox;
    int classID;
    int Probability;
};

class Detector
{
public:
    Detector(float DetectionThreshold = 0.5);
    std::vector<Detection> Forward(cv::Mat &img);
    void SetThreshold(float thresh);
    float GetThreshold();
    int init(const std::string& modelFilepath);
private:

    float threshold = 0.5;
    Ort::Session *session;
    Ort::Env *env;
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    std::vector<Ort::Value> inputTensors;
    std::vector<std::vector<int64_t>> outputDims;


};

#endif // Detector_H
