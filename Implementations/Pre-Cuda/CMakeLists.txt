cmake_minimum_required(VERSION 3.10)
project(Pre-Cuda)

set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )

add_executable(FeatureExtractor main.cpp
        Window.cpp Window.h
        GrayPair.cpp GrayPair.h
        AggregatedGrayPair.cpp AggregatedGrayPair.h
        GLCM.cpp GLCM.h
        ImageData.cpp ImageData.h
        Features.cpp Features.h
        Direction.cpp Direction.h
        FeatureComputer.cpp FeatureComputer.h
        WindowFeatureComputer.cpp WindowFeatureComputer.h
        ImageFeatureComputer.cpp ImageFeatureComputer.h
        ImageLoader.cpp ImageLoader.h
        Image.cpp Image.h WorkArea.cpp WorkArea.h)

target_link_libraries(FeatureExtractor ${OpenCV_LIBS})
