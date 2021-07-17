#include "detector.h"




Detector::Detector(float DetectionThreshold)
{

    threshold = DetectionThreshold;
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
int Detector::init(const std::string& modelFilepath)
{


    std::string instanceName{"ssd-inference"};
    env = new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                 instanceName.c_str());
    Ort::SessionOptions sessionOptions;

    bool CUDA = true;

    if(CUDA){
        OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
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

    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();


    inputNames.push_back(inputName);




    for(int n=0;n<numOutputNodes;n++){
        const char* outputName = session->GetOutputName(n, allocator);

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


    std::vector<int64_t>inputDims = {1,H,W,3};
    auto inputTensorSize = H*W*3;
    inputTensors.clear();
    inputTensors.push_back(Ort::Value::CreateTensor<uint8_t>(
        memoryInfo, img.data, inputTensorSize, inputDims.data(),
        inputDims.size()));

    auto outputTensors = session->Run(Ort::RunOptions{nullptr}, inputNames.data(),
                        inputTensors.data(), 1, outputNames.data(), outputNames.size());


     std::vector<Detection> results;
    int num_detections = outputTensors[3].GetTensorData<float>()[0];
    for(int i=0;i<num_detections;i++){
        auto data = &outputTensors[0].GetTensorData<float>()[4*i];
        int cls = outputTensors[1].GetTensorData<float>()[i];
        float score = outputTensors[2].GetTensorData<float>()[i];
        if(score<threshold)
            continue;

        float xmin = data[1];
        float xmax = data[3];
        float ymin = data[0];
        float ymax = data[2];
        cv::Rect2f box(xmin,ymin,xmax-xmin,ymax-ymin);
        Detection det;
        det.bbox = box;
        det.classID = cls;
        det.Probability = score;
        results.emplace_back(det);
    }

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
