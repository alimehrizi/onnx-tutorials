cmake_minimum_required(VERSION 3.10)

project(onnx2-detector)
find_package(Boost COMPONENTS system filesystem  REQUIRED)
find_package(OpenCV PATHS ~/OPENCV-4.5.0 NO_DEFAULT_PATH)
include_directories(/home/altex/ONNX-LIBS/new_build/include)
include_directories(/home/altex/ONNX-LIBS/new_build/include/core/session)
aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})

target_link_libraries(${PROJECT_NAME} /home/altex/ONNX-LIBS/new_build/libs/RelWithDebInfo/libonnxruntime.so
    /home/altex/ONNX-LIBS/new_build/libs/RelWithDebInfo/libonnxruntime_providers_openvino.so
    /home/altex/ONNX-LIBS/new_build/libs/RelWithDebInfo/libonnxruntime_providers_shared.so
    )
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES})



