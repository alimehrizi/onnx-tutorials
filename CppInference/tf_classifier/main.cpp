

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include"classifier.h"

int main(int argc, char* argv[])
{

    std::string modelFilepath{"/media/altex/XcDrive/konnect/OCR/1/PlateRecognition_R40_C160.out"};
    std::string imageFilepath{
        "/home/altex/Pictures/test.jpg"};

    Classifier plate_classifier;

    plate_classifier.init(modelFilepath);
    cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);
    int N = 1000;
    std::chrono::steady_clock::time_point start1 = std::chrono::steady_clock::now();
    for(int i=0;i<N;i++)
        plate_classifier.Forward(imageBGR);
    std::chrono::steady_clock::time_point stop2 = std::chrono::steady_clock::now();
    float time = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start1).count()/(float)1000;
    std::cout << "Avg time: " << time/N << " ms\n";



}
