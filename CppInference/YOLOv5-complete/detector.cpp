#include "detector.h"

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

std::vector<float> Detector::LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& new_size) {
    auto h0 = static_cast<float>(src.rows);
    auto w0 = static_cast<float>(src.cols);
    float out_h = new_size.height;
    float out_w = new_size.width;

    float r = std::min(out_w / w0, out_h / h0);
    auto interp = cv::INTER_AREA;
    if(r>1)
        interp = cv::INTER_LINEAR;
    int h = r*h0;
    int w = r*w0;
    if(r>1 | r<1)
        cv::resize(src, dst, cv::Size(w, h),0,0,interp);

    int new_unpad_w = w;
    int new_unpad_h = h;
    float dh = out_h - new_unpad_h;
    float dw = out_w - new_unpad_w;
    dw /= 2;
    dh /= 2;

    int top = round(dh-0.1);
    int bot = round(dh+0.1);
    int left = round(dw-0.1);
    int right = round(dw+0.1);

    cv::copyMakeBorder(dst, dst, top, bot, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    std::vector<float> pad_info{h0,w0, float(h/h0), float(w/w0), dw, dh};
    return pad_info;
}


Detector::Detector(float DetectionThreshold)
{

    threshold = DetectionThreshold;
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
int Detector::init(const std::string& modelFilepath)
{


    std::string instanceName{"yolo-inference"};
    env = new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                 instanceName.c_str());
    Ort::SessionOptions sessionOptions;

    bool CUDA = false;

    if(CUDA){
        CUDA = false;
    }else{

        bool enable_cpu_mem_arena = true;
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(sessionOptions, enable_cpu_mem_arena));
    }
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    session = new Ort::Session(*env, modelFilepath.c_str(), sessionOptions);


    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session->GetInputCount();
    size_t numOutputNodes = session->GetOutputCount();



    const char* inputName = session->GetInputName(0, allocator);


    Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();

    std::vector<int64_t> inputDim = inputTensorInfo.GetShape();

    inputDims.push_back(inputDim);
    inputNames.push_back(inputName);



    numOutputNodes = 1;
    for(int n=0;n<numOutputNodes;n++){
        const char* outputName = session->GetOutputName(n, allocator);
        std::cout<<outputName<<std::endl;

        outputNames.push_back(outputName);
        Ort::TypeInfo outputTypeInfo = session->GetOutputTypeInfo(n);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();


        std::vector<int64_t> outputDim = outputTensorInfo.GetShape();

        outputDims.push_back(outputDim);

    }








    return 0;
}

// Run the image through the model.
std::vector<Detection> Detector::Forward(cv::Mat &img)
{
    int H = img.size().height;
    int W = img.size().width;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    cv::Mat resizedImageBGR;
    cv::Mat resizedImage;
    cv::Mat preprocessedImage;
    std::vector<int64_t> inputDim = {1,3, 640, 640};
    auto pad_info = LetterboxImage(img,resizedImageBGR, cv::Size(inputDim.at(2), inputDim.at(3)) );
    resizedImageBGR.convertTo(resizedImage, CV_32F, 1.0 / 255.0);

    // HWC to CHW
    cv::dnn::blobFromImage(resizedImage, preprocessedImage);

    float *input_data = (float*)malloc(vectorProduct(inputDim)*sizeof(float));
    size_t inputTensorSize = vectorProduct(inputDim);
    inputTensors.clear();
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, input_data, inputTensorSize, inputDim.data(),
            inputDim.size()));

    memcpy(input_data, preprocessedImage.data,vectorProduct(inputDim)*sizeof(float));




    auto outputTensors = session->Run(Ort::RunOptions{nullptr}, inputNames.data(),
                        inputTensors.data(), 1, outputNames.data(), outputNames.size());



    auto results = postProcessCpu(outputTensors[0].GetTensorData<float>(),outputDims[0],threshold,0.6, pad_info);

    return results;
}

float Detector::GetThreshold()
{
    return threshold;
}



void Detector::SetThreshold(float thresh)
{
    if(thresh > 1.0)
        thresh = 1.0;

    if(thresh < 0)
        thresh = 0;

    threshold = thresh;
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



std::vector<Detection> Detector::postProcessCpu(const float *result,std::vector<int64_t> outputDims, float conf_thresh, float iou_thresh, std::vector<float> pad_info){
    std::vector<Detection> results;
    for(int b=0;b<outputDims[0];b++){
        std::vector<float> scores;
        std::vector<int> classes;
        std::vector<cv::Rect> bboxes;
        std::vector<cv::Rect> bboxes_offsets;
        int num_classes = outputDims[2] - 5;
        for(int i=0;i<outputDims[1]*outputDims[2];i+=outputDims[2]){
            float conf = result[i+4];

            if(conf<conf_thresh)
                continue;
//            std::cout<<conf<<std::endl;
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

        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(bboxes_offsets, scores, conf_thresh, iou_thresh, nms_indices);
        float scale = pad_info[2];
        int pad_w = pad_info[4];
        int pad_h = pad_info[5];

        for(int idx:nms_indices){
            cv::Rect2f box = bboxes[idx];
            box.x = (box.x-pad_w)/ scale;
            box.y = (box.y - pad_h)/scale;
            box.width /= scale;
            box.height /= scale;

            Detection det;
            det.bbox = box;
            det.classID = classes[idx];
            det.Probability = scores[idx];
            results.push_back(det);
        }
    }
    return results;
}
