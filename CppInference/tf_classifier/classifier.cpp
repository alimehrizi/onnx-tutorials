#include "classifier.h"

Classifier::Classifier()
{


}

int Classifier::init(std::string modelFilepath){

    std::string instanceName{"classifier-inference"};
    env = new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                 instanceName.c_str());
    Ort::SessionOptions sessionOptions;



    bool enable_cpu_mem_arena = true;
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(sessionOptions, enable_cpu_mem_arena));

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

    inputDims = inputTensorInfo.GetShape();
    inputDims[0] = 1;


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
Classifier::TensorFlowDNNOutputStruct Classifier::Forward(cv::Mat &img)
{

    cv::Mat resizedImageGray;
    cv::resize(img,resizedImageGray,cv::Size(inputDims[2],inputDims[1]));
    if(img.channels()==3)
    {
        cv::cvtColor(resizedImageGray,resizedImageGray,CV_BGR2GRAY);
    }
    std::vector<float> inputTensorValues(1);

    resizedImageGray.convertTo(resizedImageGray, CV_32FC1,1.0/255.0);
    inputTensorValues.assign(resizedImageGray.begin<float>(),
                             resizedImageGray.end<float>());


    int inputTensorSize = inputDims[0] * inputDims[1] * inputDims[2];

    std::vector<Ort::Value> inputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
        inputDims.size()));


    // https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_cxx_api.h#L353
    auto outputTensors = session->Run(Ort::RunOptions{nullptr}, inputNames.data(),
                inputTensors.data(), 1, outputNames.data(),
                 outputNames.size());
    TensorFlowDNNOutputStruct DNNOS;
    for (int i = 0; i < outputDims.size(); ++i)
    {
        auto Scores = outputTensors[i].GetTensorData<float>();
        float MaxScore = Scores[0];
        int MaxIdx = 0;

        for (int j = 0; j < outputDims[i][1]; j++)
        {
            if (Scores[j] > MaxScore)
            {
                MaxScore = Scores[j];
                MaxIdx = j;
            }
        }

        if(i==2)//alphabet
        {
            if(MaxIdx<9)
            {
                DNNOS.PlateValue.append(std::to_string(0));
            }
            DNNOS.PlateValue.append(std::to_string(MaxIdx+1));
        }
        else if(i<7)//other numbers
        {
            DNNOS.PlateValue.append(std::to_string(MaxIdx+1));
        }
        else //last number
        {
            if(MaxIdx==9)
            {
                DNNOS.PlateValue.append(std::to_string(0));
            }
            else
            {
                DNNOS.PlateValue.append(std::to_string(MaxIdx+1));
            }
        }
        DNNOS.ProbabilityVec.push_back((int)100*MaxScore);
    }





    return DNNOS;

}

