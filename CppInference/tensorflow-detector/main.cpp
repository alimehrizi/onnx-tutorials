 #include "detector.h"
#include <boost/filesystem.hpp>
#include"extra_utils.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace std;
using namespace cv;



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

int main()
{
    cout << "opencv version: "<< CV_VERSION <<endl;
    string modelFilepath = "/media/altex/XcDrive/konnect/OCR/1/PlateDetection_R150_C150.out";
    std::string imageFilepath =   "/media/altex/XcDrive/konnect/OCR/Plate/2020-09-15_00-06-05_671051572_39_2_1_L2_P.jpg";
     std::string saveResultPath = "/media/altex/XcDrive/konnect/OCR/tfdetector-result";
    create_directory(saveResultPath);

    Detector plate_detector(0.3);
    plate_detector.init(modelFilepath);

    cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);

    int N = 1000;
    std::chrono::steady_clock::time_point start1 = std::chrono::steady_clock::now();
    for(int i=0;i<N;i++)
        plate_detector.Forward(imageBGR);
    std::chrono::steady_clock::time_point stop2 = std::chrono::steady_clock::now();
    float time = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start1).count()/(float)1000;
    std::cout << "Avg time: " << time/N << " ms\n";



    return 0;
}
