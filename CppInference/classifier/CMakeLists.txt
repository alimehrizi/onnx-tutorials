cmake_minimum_required(VERSION 2.8)

project(inference)
find_package(OpenCV PATHS ~/OPENCV-4.5.0 NO_DEFAULT_PATH)
include_directories(/home/altex/ONNX/1.7/build/native/include)
add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME} /home/altex/ONNX/1.7/runtimes/linux-x64/native/libonnxruntime.so
                                        ${OpenCV_LIBS})
