 #include "detector.h"
#include <boost/filesystem.hpp>
#include"extra_utils.h"
#include"visualizer.h"
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
    std::string graph_path = "/home/altex/Models-custome1/TODv1.1-yolov5s-800-multiScale/exp/weights/best.onnx";
    std::string imageFolderPath =   "/home/altex/test_images/images2";
    std::string saveResultPath = "/home/altex/onnxCar-result";
    create_directory(saveResultPath);

    Detector detector(0.1);
    detector.init(graph_path);
    int nF = 0;
    Visualizer box_visualizer;
    //////////
    std::vector<std::vector<Detection>> results;



    std::vector<cv::Mat> images;
    std::vector<std::string> names;
    std::vector<float> pad_info;
    float t_p=0;
    Timer timer;
    int n_images=0;
    int batch_size = 2;
    std::vector<cv::Mat> batch_images;
    for (auto & entry : boost::filesystem::directory_iterator(imageFolderPath)){
        std::string img_path = entry.path().string();
        if(n_images>100)break;
        auto img = cv::imread(img_path);
        if(img.empty())continue;

         names.push_back(img_path);
         batch_images.push_back(img);

         //std::cout<<img_path<<std::endl;



        if(batch_images.size()==batch_size){
            timer.Start();
            auto preds = detector.Forward(batch_images,4);

            t_p += timer.TimeSpent();
            for(int i=0;i<batch_images.size();i++){
                cv::Mat img = batch_images[i];
                auto W = img.size().width;
                auto H = img.size().height;
                for(auto p:preds[i]){
                    auto box = p.bbox;
        //            box.x *= W;
        //            box.y *= H;
        //            box.width *= W;
        //            box.height *= H;
                    cv::rectangle(img,box,cv::Scalar(255),1);
                }
//                cv::imshow("result", img);
//                cv::waitKey(0);
                results.push_back(preds[i]);
                n_images++;
            }
            batch_images.clear();

        }





    }


    std::cout<<"graph average running time = "<<t_p/names.size()<< "ms "<<std::endl;

    for(int b=0;b<results.size();b++){
        std::ofstream result_file;
        std::string image_name = split(split(names[b],"/").back(),".")[0];
        std::string result_path = saveResultPath + "/"+image_name+".txt";
        //std::cout<<result_path<<std::endl;
        result_file.open(result_path,std::ios::out);

        for(int i=0;i<results[b].size();i++){
            auto box = results[b][i].bbox;
            int cls = results[b][i].classID;
            float score = results[b][i].Probability;
            if(cls==3)
                cls=0;
            else if(cls==2)
                cls=1;
            else
                cls=2;
            result_file<<box.x<<" "<<box.y<<" "<<box.width<<" "<<box.height<<" "<<score<<" "<<cls<<" \n";
            //result_file<<cls<<" "<<box.x+box.width/2<<" "<<box.y+box.height/2<<" "<<box.width<<" "<<box.height<<"\n";
        }
        result_file.close();
    }



    return 0;
}
