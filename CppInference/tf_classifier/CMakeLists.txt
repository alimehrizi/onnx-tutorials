cmake_minimum_required(VERSION 2.8)

project(classifier)
find_package(OpenCV PATHS ~/OPENCV-4.5.0 NO_DEFAULT_PATH)
include_directories(/home/altex/ONNX-LIBS/cpu/1.7/build/native/include)
find_package(Boost COMPONENTS system filesystem  REQUIRED)
aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} /home/altex/ONNX-LIBS/cpu/1.7/runtimes/linux-x64/native/libonnxruntime.so
                                        ${OpenCV_LIBS}
                                         ${Boost_LIBRARIES})
