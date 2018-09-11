#ifndef CUDAFUNCTIONS_H_
#define CUDAFUNCTIONS_H_

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__ 
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif

#include <iostream>

#include "Window.h"
#include "ImageData.h"

using namespace std;

/*
 * Error handling
 */
void cudaCheckError(cudaError_t err);
void checkKernelLaunchError();
/*
 * Querying info
 */
void printGPULaunchConfiguration(dim3 Grid, dim3 Blocks);
void queryGPUData();
/*
 * Block dimensioning
 */
int getCudaBlockSideX();
int getCudaBlockSideY();
dim3 getBlockConfiguration();
/*
 * Grid dimensioning
 */
int getGridSide(int imageRows, int imageCols);
dim3 getGridFromImage(int imageRows, int imageCols);
dim3 getGridFromAvailableMemory(int numberOfPairs,
 size_t featureSize);
dim3 getGrid(int numberOfPairsInWindow, size_t featureSize, int imgRows, int imgCols);
__device__ WorkArea generateThreadWorkArea(int numberOfPairs, 
	double* d_featuresList);
/*
 * Memory checks
 */
void incrementGPUHeap(size_t newHeapSize, size_t featureSize);
void handleInsufficientMemory();
bool checkEnoughWorkingAreaForThreads(int numberOfPairs, int numberOfThreads,
 size_t featureSize);

/*
 * Kernel 
 */
__global__ void computeFeatures(unsigned int * pixels, 
	ImageData img, Window windowData, int numberOfPairsInWindow, 
	double* d_featuresList);
#endif /* FEATURECOMPUTER_H_ */
