# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/cmake/936/bin/cmake

# The command to remove a file.
RM = /snap/cmake/936/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/altex/onnx-tutorials/CppInference/YOLOv5-complete-batchMode

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/altex/onnx-tutorials/CppInference/build-YOLOv5-complete-Desktop-Default

# Include any dependencies generated for this target.
include CMakeFiles/onnx-detector.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/onnx-detector.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/onnx-detector.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/onnx-detector.dir/flags.make

CMakeFiles/onnx-detector.dir/detector.cpp.o: CMakeFiles/onnx-detector.dir/flags.make
CMakeFiles/onnx-detector.dir/detector.cpp.o: /home/altex/onnx-tutorials/CppInference/YOLOv5-complete-batchMode/detector.cpp
CMakeFiles/onnx-detector.dir/detector.cpp.o: CMakeFiles/onnx-detector.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/altex/onnx-tutorials/CppInference/build-YOLOv5-complete-Desktop-Default/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/onnx-detector.dir/detector.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/onnx-detector.dir/detector.cpp.o -MF CMakeFiles/onnx-detector.dir/detector.cpp.o.d -o CMakeFiles/onnx-detector.dir/detector.cpp.o -c /home/altex/onnx-tutorials/CppInference/YOLOv5-complete-batchMode/detector.cpp

CMakeFiles/onnx-detector.dir/detector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/onnx-detector.dir/detector.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/altex/onnx-tutorials/CppInference/YOLOv5-complete-batchMode/detector.cpp > CMakeFiles/onnx-detector.dir/detector.cpp.i

CMakeFiles/onnx-detector.dir/detector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/onnx-detector.dir/detector.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/altex/onnx-tutorials/CppInference/YOLOv5-complete-batchMode/detector.cpp -o CMakeFiles/onnx-detector.dir/detector.cpp.s

CMakeFiles/onnx-detector.dir/extra_utils.cpp.o: CMakeFiles/onnx-detector.dir/flags.make
CMakeFiles/onnx-detector.dir/extra_utils.cpp.o: /home/altex/onnx-tutorials/CppInference/YOLOv5-complete-batchMode/extra_utils.cpp
CMakeFiles/onnx-detector.dir/extra_utils.cpp.o: CMakeFiles/onnx-detector.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/altex/onnx-tutorials/CppInference/build-YOLOv5-complete-Desktop-Default/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/onnx-detector.dir/extra_utils.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/onnx-detector.dir/extra_utils.cpp.o -MF CMakeFiles/onnx-detector.dir/extra_utils.cpp.o.d -o CMakeFiles/onnx-detector.dir/extra_utils.cpp.o -c /home/altex/onnx-tutorials/CppInference/YOLOv5-complete-batchMode/extra_utils.cpp

CMakeFiles/onnx-detector.dir/extra_utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/onnx-detector.dir/extra_utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/altex/onnx-tutorials/CppInference/YOLOv5-complete-batchMode/extra_utils.cpp > CMakeFiles/onnx-detector.dir/extra_utils.cpp.i

CMakeFiles/onnx-detector.dir/extra_utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/onnx-detector.dir/extra_utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/altex/onnx-tutorials/CppInference/YOLOv5-complete-batchMode/extra_utils.cpp -o CMakeFiles/onnx-detector.dir/extra_utils.cpp.s

CMakeFiles/onnx-detector.dir/main.cpp.o: CMakeFiles/onnx-detector.dir/flags.make
CMakeFiles/onnx-detector.dir/main.cpp.o: /home/altex/onnx-tutorials/CppInference/YOLOv5-complete-batchMode/main.cpp
CMakeFiles/onnx-detector.dir/main.cpp.o: CMakeFiles/onnx-detector.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/altex/onnx-tutorials/CppInference/build-YOLOv5-complete-Desktop-Default/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/onnx-detector.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/onnx-detector.dir/main.cpp.o -MF CMakeFiles/onnx-detector.dir/main.cpp.o.d -o CMakeFiles/onnx-detector.dir/main.cpp.o -c /home/altex/onnx-tutorials/CppInference/YOLOv5-complete-batchMode/main.cpp

CMakeFiles/onnx-detector.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/onnx-detector.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/altex/onnx-tutorials/CppInference/YOLOv5-complete-batchMode/main.cpp > CMakeFiles/onnx-detector.dir/main.cpp.i

CMakeFiles/onnx-detector.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/onnx-detector.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/altex/onnx-tutorials/CppInference/YOLOv5-complete-batchMode/main.cpp -o CMakeFiles/onnx-detector.dir/main.cpp.s

CMakeFiles/onnx-detector.dir/visualizer.cpp.o: CMakeFiles/onnx-detector.dir/flags.make
CMakeFiles/onnx-detector.dir/visualizer.cpp.o: /home/altex/onnx-tutorials/CppInference/YOLOv5-complete-batchMode/visualizer.cpp
CMakeFiles/onnx-detector.dir/visualizer.cpp.o: CMakeFiles/onnx-detector.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/altex/onnx-tutorials/CppInference/build-YOLOv5-complete-Desktop-Default/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/onnx-detector.dir/visualizer.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/onnx-detector.dir/visualizer.cpp.o -MF CMakeFiles/onnx-detector.dir/visualizer.cpp.o.d -o CMakeFiles/onnx-detector.dir/visualizer.cpp.o -c /home/altex/onnx-tutorials/CppInference/YOLOv5-complete-batchMode/visualizer.cpp

CMakeFiles/onnx-detector.dir/visualizer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/onnx-detector.dir/visualizer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/altex/onnx-tutorials/CppInference/YOLOv5-complete-batchMode/visualizer.cpp > CMakeFiles/onnx-detector.dir/visualizer.cpp.i

CMakeFiles/onnx-detector.dir/visualizer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/onnx-detector.dir/visualizer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/altex/onnx-tutorials/CppInference/YOLOv5-complete-batchMode/visualizer.cpp -o CMakeFiles/onnx-detector.dir/visualizer.cpp.s

# Object files for target onnx-detector
onnx__detector_OBJECTS = \
"CMakeFiles/onnx-detector.dir/detector.cpp.o" \
"CMakeFiles/onnx-detector.dir/extra_utils.cpp.o" \
"CMakeFiles/onnx-detector.dir/main.cpp.o" \
"CMakeFiles/onnx-detector.dir/visualizer.cpp.o"

# External object files for target onnx-detector
onnx__detector_EXTERNAL_OBJECTS =

onnx-detector: CMakeFiles/onnx-detector.dir/detector.cpp.o
onnx-detector: CMakeFiles/onnx-detector.dir/extra_utils.cpp.o
onnx-detector: CMakeFiles/onnx-detector.dir/main.cpp.o
onnx-detector: CMakeFiles/onnx-detector.dir/visualizer.cpp.o
onnx-detector: CMakeFiles/onnx-detector.dir/build.make
onnx-detector: /home/altex/ONNX-LIBS/onnxruntime-linux-x64-1.8.1/lib/libonnxruntime.so
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_gapi.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_stitching.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_alphamat.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_aruco.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_bgsegm.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_bioinspired.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_ccalib.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_cudabgsegm.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_cudafeatures2d.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_cudaobjdetect.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_cudastereo.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_cvv.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_dnn_objdetect.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_dnn_superres.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_dpm.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_face.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_freetype.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_fuzzy.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_hdf.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_hfs.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_img_hash.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_intensity_transform.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_line_descriptor.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_mcc.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_quality.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_rapid.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_reg.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_rgbd.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_saliency.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_sfm.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_stereo.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_structured_light.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_superres.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_surface_matching.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_tracking.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_videostab.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_xfeatures2d.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_xobjdetect.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_xphoto.so.4.5.0
onnx-detector: /usr/lib/x86_64-linux-gnu/libboost_system.so
onnx-detector: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_highgui.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_shape.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_datasets.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_plot.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_text.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_dnn.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_ml.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_phase_unwrapping.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_cudacodec.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_videoio.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_cudaoptflow.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_cudalegacy.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_cudawarping.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_optflow.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_ximgproc.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_video.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_imgcodecs.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_objdetect.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_calib3d.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_features2d.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_flann.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_photo.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_cudaimgproc.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_cudafilters.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_imgproc.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_cudaarithm.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_core.so.4.5.0
onnx-detector: /home/altex/OPENCV-4.5.0/lib/libopencv_cudev.so.4.5.0
onnx-detector: CMakeFiles/onnx-detector.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/altex/onnx-tutorials/CppInference/build-YOLOv5-complete-Desktop-Default/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable onnx-detector"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/onnx-detector.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/onnx-detector.dir/build: onnx-detector
.PHONY : CMakeFiles/onnx-detector.dir/build

CMakeFiles/onnx-detector.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/onnx-detector.dir/cmake_clean.cmake
.PHONY : CMakeFiles/onnx-detector.dir/clean

CMakeFiles/onnx-detector.dir/depend:
	cd /home/altex/onnx-tutorials/CppInference/build-YOLOv5-complete-Desktop-Default && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/altex/onnx-tutorials/CppInference/YOLOv5-complete-batchMode /home/altex/onnx-tutorials/CppInference/YOLOv5-complete-batchMode /home/altex/onnx-tutorials/CppInference/build-YOLOv5-complete-Desktop-Default /home/altex/onnx-tutorials/CppInference/build-YOLOv5-complete-Desktop-Default /home/altex/onnx-tutorials/CppInference/build-YOLOv5-complete-Desktop-Default/CMakeFiles/onnx-detector.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/onnx-detector.dir/depend

