cmake_minimum_required(VERSION 2.8)

project( FeaturesExtractor )

find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(FeaturesExtractor FeatureExtractor.cpp )
target_link_libraries( FeaturesExtractor ${OpenCV_LIBS} )

