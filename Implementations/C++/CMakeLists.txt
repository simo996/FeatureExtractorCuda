cmake_minimum_required(VERSION 3.10)
project(FeatureExtractor)

set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )

add_executable(FeatureExtractor main.cpp
        Window.h Window.cpp
        GrayPair.cpp GrayPair.h
        AggregatedGrayPair.cpp AggregatedGrayPair.h
        GLCM.cpp GLCM.h
        FeatureComputer.cpp FeatureComputer.h
        WindowFeatureComputer.cpp WindowFeatureComputer.h
        ImageFeatureComputer.cpp ImageFeatureComputer.h Image.cpp Image.h Features.cpp Features.h Direction.cpp Direction.h ImageLoader.cpp ImageLoader.h)

target_link_libraries(FeatureExtractor ${OpenCV_LIBS})
