cmake_minimum_required(VERSION 3.10 FATAL_ERROR )
project(Pre-Cuda LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)

# Root dir
set(PROJECT_ROOT_DIR ${PROJECT_SOURCE_DIR})

# Binaries directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_ROOT_DIR}/bin)

# Update sources directory
set(PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

# Include directories
include_directories(
        ${PROJECT_SOURCE_DIR}
)

add_executable(FeatureExtractor
        ${PROJECT_SOURCE_DIR}/main.cpp

        ${PROJECT_SOURCE_DIR}/Window.cpp
        ${PROJECT_SOURCE_DIR}/Window.h

        ${PROJECT_SOURCE_DIR}/GrayPair.cpp
        ${PROJECT_SOURCE_DIR}/GrayPair.h

        ${PROJECT_SOURCE_DIR}/AggregatedGrayPair.cpp
        ${PROJECT_SOURCE_DIR}/AggregatedGrayPair.h

        ${PROJECT_SOURCE_DIR}/GLCM.cpp
        ${PROJECT_SOURCE_DIR}/GLCM.h

        ${PROJECT_SOURCE_DIR}/ImageData.cpp
        ${PROJECT_SOURCE_DIR}/ImageData.h

        ${PROJECT_SOURCE_DIR}/Features.cpp
        ${PROJECT_SOURCE_DIR}/Features.h

        ${PROJECT_SOURCE_DIR}/Direction.cpp
        ${PROJECT_SOURCE_DIR}/Direction.h

        ${PROJECT_SOURCE_DIR}/FeatureComputer.cpp
        ${PROJECT_SOURCE_DIR}/FeatureComputer.h

        ${PROJECT_SOURCE_DIR}/WindowFeatureComputer.cpp
        ${PROJECT_SOURCE_DIR}/WindowFeatureComputer.h

        ${PROJECT_SOURCE_DIR}/ImageFeatureComputer.cpp
        ${PROJECT_SOURCE_DIR}/ImageFeatureComputer.h

        ${PROJECT_SOURCE_DIR}/ImageLoader.cpp
        ${PROJECT_SOURCE_DIR}/ImageLoader.h

        ${PROJECT_SOURCE_DIR}/Image.cpp
        ${PROJECT_SOURCE_DIR}/Image.h

        ${PROJECT_SOURCE_DIR}/WorkArea.cpp
        ${PROJECT_SOURCE_DIR}/WorkArea.h src/ProgramArguments.cpp src/ProgramArguments.h)

# Add OpenCvTo the project
find_package( OpenCV REQUIRED )
target_link_libraries(FeatureExtractor ${OpenCV_LIBS})