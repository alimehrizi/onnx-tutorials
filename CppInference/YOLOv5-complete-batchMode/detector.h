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
#include <opencv2/dnn/dnn.hpp>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

struct Detection
{
    float time;
    cv::Rect2f bbox;
    int classID;
    float Probability;
};

class Detector
{
public:
    Detector(float DetectionThreshold = 0.5);
    std::vector<std::vector<Detection> > Forward(std::vector<cv::Mat> &images, int num_classes=-1);
    void SetThreshold(float thresh);
    float GetThreshold();
    int init(const std::string& modelFilepath);
private:
    std::vector<Detection> postProcessCpu(const float *result, std::vector<int64_t> outputDims, float conf_thresh, float iou_thresh, std::vector<float> pad_info);
    std::vector<float> LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& new_size);
    float threshold = 0.5;
    Ort::Session *session;
    Ort::Env *env;
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    std::vector<Ort::Value> inputTensors;
    std::vector<std::vector<int64_t>> outputDims;
    std::vector<std::vector<int64_t>> inputDims;


};

#endif // Detector_H
