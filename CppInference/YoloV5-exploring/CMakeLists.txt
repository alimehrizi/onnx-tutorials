cmake_minimum_required(VERSION 2.8)

project(onnx-detector)
find_package(OpenCV PATHS ~/OPENCV-4.5.0 NO_DEFAULT_PATH)
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda-11.0/include)
link_directories(/usr/local/cuda-11.0/lib64)
include_directories(/home/altex/ONNX-LIBS/cuda/1.7/build/native/include)

cuda_add_library(cuda_kernels SHARED  "cuda_utils.h" "cuda_kernels.cu")
target_link_libraries(cuda_kernels ${CUDA_LIBRARIES} )
find_package(Boost COMPONENTS system filesystem  REQUIRED)

add_executable(${PROJECT_NAME} "main.cpp" "extra_utils.cpp" "extra_utils.h")
target_link_libraries(${PROJECT_NAME} /home/altex/ONNX-LIBS/cuda/1.7/runtimes/linux-x64/native/libonnxruntime.so
                                        ${OpenCV_LIBS}
                                        ${Boost_LIBRARIES}
                                        cuda_kernels)
