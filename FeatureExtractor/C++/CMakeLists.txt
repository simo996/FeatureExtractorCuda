cmake_minimum_required(VERSION 3.10)
project(FeatureExtractor)

set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )

add_executable(FeatureExtractor main.cpp GrayPair.cpp GrayPair.h AggregatedGrayPair.cpp AggregatedGrayPair.h GLCM.cpp GLCM.h WindowFeatureComputer.cpp WindowFeatureComputer.h)