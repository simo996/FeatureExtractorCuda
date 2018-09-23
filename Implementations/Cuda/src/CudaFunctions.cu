#include "CudaFunctions.h"

/* CUDA METHODS */

/**
 * Exit the program if any of the cuda runtime function invocation controlled
 * with this method returns a failure
 */
void cudaCheckError(cudaError_t err){
	if( err != cudaSuccess ) {
		cerr << "ERROR: " << cudaGetErrorString(err) << endl;
		exit(-1);
	}
}

 
/**
 * Exit the program if the launch of the only kernel of computation fails
 * Need to call after kernel invocation 
 */
void checkKernelLaunchError(){
	cudaError_t errSync  = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess) // Detect configuration launch errors
		printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess) // Detect kernel execution errors
		printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}

/**
 * Print how many blocks and how many threads per block will compute the 
 * the kernel
 */
void printGPULaunchConfiguration(dim3 Grid, dim3 Blocks){
	cout << "\t- GPU Launch Configuration -" << endl;
	cout << "\t GRID\t rows: " << Grid.y << " x cols: " << Grid.x << endl;
	cout << "\t BLOCK\t rows: " << Blocks.y << " x cols: " << Blocks.x << endl;
}

/*
 * Querying and print info on the gpu of the system
 */
void queryGPUData(){
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	cout << "\t- GPU DATA -" << endl;
	cout << "\tDevice name: " << prop.name << endl;
	cout << "\tNumber of multiprocessors: " << prop.multiProcessorCount << endl;
	size_t gpuMemory = prop.totalGlobalMem;
	cout << "\tTotalGlobalMemory: " << (gpuMemory / 1024 / 1024) << " MB" << endl;
}

 /* 
	Default Cuda Block Dimensioning 
	Square with side of 12
*/
int getCudaBlockSideX(){
	return 12;
}

// Square
int getCudaBlockSideY(){
	return getCudaBlockSideX();
}

/* 
 * The block is always fixed, only the grid changes 
 * according to gpu memory/image size 
*/
dim3 getBlockConfiguration()
{
	// TODO implement GPU architecture specific dimensioning 
	int ROWS = getCudaBlockSideY();
	int COLS = getCudaBlockSideX(); 
	assert(ROWS * COLS <= 256);
	dim3 configuration(ROWS, COLS);
	return configuration;
}

/**
 * Returns the side of a grid obtained from the image physical dimension where
 * each thread computes just a window
 */
int getGridSide(int imageRows, int imageCols){
	// Smallest side of a rectangular image will determine grid dimension
	int imageSmallestSide = imageRows;
	if(imageCols < imageSmallestSide)
		imageSmallestSide = imageCols;
   
	int blockSide = getCudaBlockSideX();
	// Check if image size is low enough to fit in maximum grid
	// round up division 
	int gridSide = (imageSmallestSide + blockSide -1) / blockSide;
	// Cant' exceed 65536 blocks in grid
	if(gridSide > 256){
		gridSide = 256;
	}
	return gridSide;
}

/**
 * Create a grid from the image physical dimension where each thread  
 * computes just a window
 */
dim3 getGridFromImage(int imageRows, int imageCols)
{
	return dim3(getGridSide(imageRows, imageCols), 
		getGridSide(imageRows, imageCols));
}


/**
 * Program aborts if not even 1 block of threads can be launched for 
 * insufficient memory (very obsolete gpu)
 */
void handleInsufficientMemory(){
	cerr << "FAILURE ! Gpu doesn't have enough memory \
	to hold the results and the space needed to threads" << endl;
	cerr << "Try lowering window side and/or symmetricity "<< endl;
	exit(-1);
}  

/**
 * Method that will check if the proposed number of threads will have enough memory
 * @param numberOfPairs: number of pixel pairs that belongs to each window
 * @param featureSize: memory space consumed by the values that will be computed
 * @param numberOfThreads: how many threads the proposed grid has
 * @param verbose: print extra info on the memory consumed
 */
bool checkEnoughWorkingAreaForThreads(int numberOfPairs, int numberOfThreads,
 size_t featureSize, bool verbose){
	// Get GPU mem size
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	size_t gpuMemory = prop.totalGlobalMem;
	// Compute the memory needed for a single thread
	size_t workAreaSpace = numberOfPairs * 
		( 4 * sizeof(AggregatedGrayPair) + 1 * sizeof(GrayPair));
	// Multiply per threads number
	size_t totalWorkAreas = workAreaSpace * numberOfThreads;

	long long int difference = gpuMemory - (featureSize + totalWorkAreas);
	if(difference <= 0){
		return false;
	}
	else{
		if(verbose){
			cout << endl << "\tGPU threads space: (MB) " << totalWorkAreas / 1024 / 1024 << endl;
			size_t free, total;
			cudaMemGetInfo(&free,&total);
			cout << "\tGPU used memory: (MB) " << (((totalWorkAreas + featureSize) / 1024) / 1024) << endl;
		}
		return true;
	}
}

/**
 * Method that will generate the smallest computing grid that can fit into
 *  the GPU memory
 * @param numberOfPairs: number of pixel pairs that belongs to each window
 * @param featureSize: memory space consumed by the values that will be computed
 */
dim3 getGridFromAvailableMemory(int numberOfPairs,
 size_t featureSize){

	// Get GPU mem size
	size_t freeGpuMemory, total;
	cudaMemGetInfo(&freeGpuMemory,&total);

	// Compute the memory needed for a single thread
	size_t workAreaSpace = numberOfPairs * 
		( 4 * sizeof(AggregatedGrayPair) + 1 * sizeof(GrayPair));

	// how many thread fit into a single block
	int threadInBlock = getCudaBlockSideX() * getCudaBlockSideX();

	size_t singleBlockMemoryOccupation = workAreaSpace * threadInBlock;
	// Even 1 block can be launched
	if(freeGpuMemory <= singleBlockMemoryOccupation){
		handleInsufficientMemory(); // exit
	}

	cout << endl << "WARNING! Maximum available gpu memory consumed" << endl;
	// how many blocks can be launched
	int numberOfBlocks = freeGpuMemory / singleBlockMemoryOccupation;
	
	// Create 2d grid of blocks
	int gridSide = sqrt(numberOfBlocks);
	return dim3(gridSide, gridSide);
}


/**
 * Method that will generate the computing grid 
 * If not even 1 block can be launched the program will abort
 * @param numberOfPairsInWindow: number of pixel pairs that belongs to each window
 * @param featureSize: memory space consumed by the values that will be computed
 * @param imgRows: how many rows the image has
 * @param imgCols: how many columns the image has
 * @param verbose: print extra info on the memory consumed
 */
dim3 getGrid(int numberOfPairsInWindow, size_t featureSize, int imgRows, 
	int imgCols, bool verbose){
 	dim3 Blocks = getBlockConfiguration();
	// Generate grid from image dimensions
	dim3 Grid = getGridFromImage(imgRows, imgCols);
	// check if there is enough space on the GPU to allocate working areas
	int numberOfBlocks = Grid.x * Grid.y;
	int numberOfThreadsPerBlock = Blocks.x * Blocks.y;
	int numberOfThreads = numberOfThreadsPerBlock * numberOfBlocks;
	if(! checkEnoughWorkingAreaForThreads(numberOfPairsInWindow, 
		numberOfThreads, featureSize, verbose))
	{
		Grid = getGridFromAvailableMemory(numberOfPairsInWindow, featureSize);
		// Get the total number of threads and see if the gpu memory is sufficient
		numberOfBlocks = Grid.x * Grid.y;
		numberOfThreads = numberOfThreadsPerBlock * numberOfBlocks;
		checkEnoughWorkingAreaForThreads(numberOfPairsInWindow, 
			numberOfThreads, featureSize, verbose);
	}
	if(verbose)
		printGPULaunchConfiguration(Grid, Blocks);
	return Grid;
}

/**
 * Method that each thread invokes when it needs to know its own unique
 * index inside the kernel launch configuration
 */
__device__ int getGlobalIdx_2D()
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x)	+ threadIdx.x;
	return threadId;
}


/**
 * Method invoked by each thread to get the reference to their own memory,
 * entirely pre-allocated on the host, needed for their computation
 * @param globalWorkArea: reference to the global, allocated by host, memory
 * that each thread will use to do their job
 * @param threadId: unique thread id inside the launch configuration
 */
__device__ WorkArea adjustThreadWorkArea(WorkArea globalWorkArea, int threadId){
	// Each 1 of these data structures allow 1 thread to work
	GrayPair* grayPairs = globalWorkArea.grayPairs;
	AggregatedGrayPair* summedPairs = globalWorkArea.summedPairs;
	AggregatedGrayPair* subtractedPairs = globalWorkArea.subtractedPairs;
	AggregatedGrayPair* xMarginalPairs = globalWorkArea.xMarginalPairs;
	AggregatedGrayPair* yMarginalPairs = globalWorkArea.yMarginalPairs;

	// Thread shift for accessing its memory
	int pointerStride = globalWorkArea.numberOfElements * threadId;

	// Point to own memory
	grayPairs += pointerStride;
	summedPairs += pointerStride;
	subtractedPairs += pointerStride;
	xMarginalPairs += pointerStride;
	yMarginalPairs += pointerStride;

	WorkArea wa(globalWorkArea.numberOfElements, grayPairs, summedPairs,
				subtractedPairs, xMarginalPairs, yMarginalPairs, 
				globalWorkArea.output);
	return wa;
}

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
	ImageData img, Window windowData,  
	WorkArea globalWorkArea){

	// Memory location on which the thread will work
	int threadUniqueId = getGlobalIdx_2D();
	WorkArea wa = adjustThreadWorkArea(globalWorkArea, threadUniqueId);

	// Get X and Y starting coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x; 
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// If 1 thread need to compute more than 1 window 
	// How many shift right for reaching the next window to compute
	int colsStride =  gridDim.x * blockDim.x; 
	// How many shift down for reaching the next window to compute
	int rowsStride =  gridDim.y * blockDim.y;

	// Consider bordering and the consequent padding
	int appliedBorders = img.getBorderSize();
	int realImageRows = img.getRows() - 2 * appliedBorders;
    int realImageCols = img.getColumns() - 2 * appliedBorders;
	
	
	// Create local window information
	Window actualWindow {windowData.side, windowData.distance,
								 windowData.directionType, windowData.symmetric};
	for(int i = y; i < realImageRows; i+= rowsStride){
		for(int j = x; j < realImageCols ; j+= colsStride){
			// tell the window its relative offset (starting point) inside the image
			actualWindow.setSpacialOffsets(i + appliedBorders, j + appliedBorders);
			// Launch the computation of features on the window
			WindowFeatureComputer wfc(pixels, img, actualWindow, wa);
		}
	}
}

