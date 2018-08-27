cmake_minimum_required(VERSION 3.8 FATAL_ERROR )

project(CUFEAT LANGUAGES C CXX CUDA)

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

add_executable(CuFeat
        ${PROJECT_SOURCE_DIR}/main.cpp

        ${PROJECT_SOURCE_DIR}/GrayPair.cu
        ${PROJECT_SOURCE_DIR}/GrayPair.h

        ${PROJECT_SOURCE_DIR}/AggregatedGrayPair.cu
        ${PROJECT_SOURCE_DIR}/AggregatedGrayPair.h

        ${PROJECT_SOURCE_DIR}/Direction.cu
        ${PROJECT_SOURCE_DIR}/Direction.h

        ${PROJECT_SOURCE_DIR}/Image.cu
        ${PROJECT_SOURCE_DIR}/Image.h

        ${PROJECT_SOURCE_DIR}/ImageData.cu
        ${PROJECT_SOURCE_DIR}/ImageData.h

        ${PROJECT_SOURCE_DIR}/Window.cu
        ${PROJECT_SOURCE_DIR}/Window.h

        ${PROJECT_SOURCE_DIR}/Features.cu
        ${PROJECT_SOURCE_DIR}/Features.h

        ${PROJECT_SOURCE_DIR}/WorkArea.cu
        ${PROJECT_SOURCE_DIR}/WorkArea.h

        ${PROJECT_SOURCE_DIR}/GLCM.cu
        ${PROJECT_SOURCE_DIR}/GLCM.h

        ${PROJECT_SOURCE_DIR}/FeatureComputer.cu
        ${PROJECT_SOURCE_DIR}/FeatureComputer.h

        ${PROJECT_SOURCE_DIR}/WindowFeatureComputer.cu
        ${PROJECT_SOURCE_DIR}/WindowFeatureComputer.h

        ${PROJECT_SOURCE_DIR}/ImageFeatureComputer.cu
        ${PROJECT_SOURCE_DIR}/ImageFeatureComputer.h

        )

# Add OpenCvTo the project
find_package( OpenCV REQUIRED )
target_link_libraries(CuFeat ${OpenCV_LIBS})

# C++11
target_compile_features(CuFeat PUBLIC cxx_std_11)

# Enable separable compilation
set_target_properties(CuFeat PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# gcc 5.4 fix
SET ( CMAKE_CXX_FLAGS "-D_MWAITXINTRIN_H_INCLUDED -D_STRICT_ANSI__ -D_FORCE_INLINES")