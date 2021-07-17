#include "visualizer.h"

Visualizer::Visualizer()
{
    Colors =
    {
        cv::Scalar(0,0,102), cv::Scalar(0,0,204), cv::Scalar(51,51,255), cv::Scalar(153,153,255),
        cv::Scalar(0,51,102), cv::Scalar(0,102,204), cv::Scalar(51,153,255), cv::Scalar(153,204,255),
        cv::Scalar(0,102,102), cv::Scalar(0.204,204), cv::Scalar(51,255,255), cv::Scalar(153,255,255),
        cv::Scalar(0,102,51), cv::Scalar(0,204,102), cv::Scalar(51,255,153), cv::Scalar(153,255,204),
        cv::Scalar(51,102,0), cv::Scalar(102,204,0), cv::Scalar(153,255,51), cv::Scalar(204,255,153),
        cv::Scalar(102,102,0), cv::Scalar(204,204,0), cv::Scalar(255,255,51), cv::Scalar(255,255,153),
        cv::Scalar(102,51,0), cv::Scalar(204,102,0), cv::Scalar(255,153,51), cv::Scalar(255,204,153),
        cv::Scalar(102,0,0), cv::Scalar(204,0,0), cv::Scalar(255,51,51), cv::Scalar(255,102,102),
        cv::Scalar(102,0,51), cv::Scalar(204,0,102), cv::Scalar(255,51,153), cv::Scalar(255,153,204),
        cv::Scalar(102,0,102), cv::Scalar(204,0,204), cv::Scalar(255,51,255), cv::Scalar(255,153,255),
        cv::Scalar(51,0,102), cv::Scalar(102,0,204), cv::Scalar(153,51,255)
    };


}

void Visualizer::drawBboxes(cv::Mat &image,std::vector<cv::Rect2d>bboxes, std::vector<int>labels, std::vector<float>scores){
    for(int i=0; i<bboxes.size();i++){
        cv::Rect2d box = bboxes.at(i);
        auto color = Colors.at(0);
        cv::rectangle(image,box,color);
    }
}
