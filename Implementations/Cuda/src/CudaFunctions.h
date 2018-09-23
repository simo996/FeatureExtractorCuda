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
#include <assert.h>

#include "Window.h"
#include "WorkArea.h"
#include "WindowFeatureComputer.h"
#include "ImageData.h"

using namespace std;

// Error handling
/**
 * Exit the program if any of the cuda runtime function invocation controlled
 * with this method returns a failure
 */
void cudaCheckError(cudaError_t err);
/**
 * Exit the program if the launch of the only kernel of computation fails
 */
void checkKernelLaunchError();
/**
 * Print how many blocks and how many threads per block will compute the 
 * the kernel
 */
void printGPULaunchConfiguration(dim3 Grid, dim3 Blocks);
/*
 * Querying info on the gpu of the system
 */
void queryGPUData();

// Block dimensioning
int getCudaBlockSideX();
int getCudaBlockSideY();
/*
 * Returns the dimension of each squared block of threads of the computation
 */
dim3 getBlockConfiguration();

//Grid dimensioning
/**
 * Returns the side of a grid obtained from the image physical dimension where
 * each thread computes just a window
 */
int getGridSide(int imageRows, int imageCols);
/**
 * Create a grid from the image physical dimension where each thread  
 * computes just a window
 */
dim3 getGridFromImage(int imageRows, int imageCols);
/**
 * Method that will generate the smallest computing grid that can fit into
 *  the GPU memory
 * @param numberOfPairs: number of pixel pairs that belongs to each window
 * @param featureSize: memory space consumed by the values that will be computed
 */
dim3 getGridFromAvailableMemory(int numberOfPairs,
 size_t featureSize);
/**
 * Method that will generate the computing grid 
 * Gpu allocable heap will be changed according to the grid individuated
 * If not even 1 block can be launched the program will abort
 * @param numberOfPairsInWindow: number of pixel pairs that belongs to each window
 * @param featureSize: memory space consumed by the values that will be computed
 * @param imgRows: how many rows the image has
 * @param imgCols: how many columns the image has
 * @param verbose: print extra info on the memory consumed
 */
dim3 getGrid(int numberOfPairsInWindow, size_t featureSize, int imgRows, 
	int imgCols, bool verbose);
/**
 * Method invoked by each thread to get the reference to their own memory,
 * entirely pre-allocated on the host, needed for their computation
 * @param globalWorkArea: reference to the global, allocated by host, memory
 * that each thread will use to do their job
 * @param threadId: unique thread id inside the launch configuration
 */
__device__ WorkArea adjustThreadWorkArea(WorkArea globalWorkArea, int threadId);
/**
 * Program aborts if not even 1 block of threads can be launched for 
 * insufficient memory (very obsolete gpu)
 */
void handleInsufficientMemory();
/**
 * Method that will check if the proposed number of threads will have enough memory
 * @param numberOfPairs: number of pixel pairs that belongs to each window
 * @param featureSize: memory space consumed by the values that will be computed
 * @param numberOfThreads: how many threads the proposed grid has
 * @param verbose: print extra info on the memory consumed
 */
bool checkEnoughWorkingAreaForThreads(int numberOfPairs, int numberOfThreads,
 size_t featureSize, bool verbose);

/**
 * Kernel that will compute all the features in each window of the image. Each
 * window will be computed by a autonomous thread of the grid
 * @param pixels: pixels intensities of the image provided
 * @param img: image metadata
 * @param globalWorkArea: class that embeds pointers to the pre-allocated 
 * space that will contain the arrays of representations that each thread will
 * use to perform its computation
 */
__global__ void computeFeatures(unsigned int * pixels, 
	ImageData img, Window windowData, WorkArea globalWorkArea);
#endif 
