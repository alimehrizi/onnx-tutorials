#ifndef VISUALIZER_H
#define VISUALIZER_H
#include "header.h"

class Visualizer
{
public:
    std::vector <cv::Scalar> Colors;
    Visualizer();
    void drawBboxes(cv::Mat &image, std::vector<cv::Rect2d>bboxes, std::vector<int> labels, std::vector<float> scores);
};

#endif // VISUALIZER_H


