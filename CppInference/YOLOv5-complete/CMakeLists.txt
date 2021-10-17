cmake_minimum_required(VERSION 2.8)

project(onnx-detector)
find_package(Boost COMPONENTS system filesystem  REQUIRED)
find_package(OpenCV PATHS ~/OPENCV-4.5.0 NO_DEFAULT_PATH)
include_directories(/home/altex/ONNX-LIBS/cuda/1.7/build/native/include)
aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})

target_link_libraries(${PROJECT_NAME} /home/altex/ONNX-LIBS/cuda/1.7/runtimes/linux-x64/native/libonnxruntime.so)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES})



