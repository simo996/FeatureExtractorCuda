#include "CudaFunctions.h"


void cudaCheckError(cudaError_t err){
	if( err != cudaSuccess ) {
		cerr << "ERROR: " << cudaGetErrorString(err) << endl;
		exit(-1);
	}
}

// Print data about kernel launch configuration
void printGPULaunchConfiguration(dim3 Grid, dim3 Blocks){
	cout << "\t- GPU Launch Configuration -" << endl;
	cout << "\t GRID\t rows: " << Grid.y << " x cols: " << Grid.x << endl;
	cout << "\t BLOCK\t rows: " << Blocks.y << " x cols: " << Blocks.x << endl;
}

// 	Print data about the GPU
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
	The block is always fixed, only the grid changes 
	according to memory/image size 
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

// create a grid from image size
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

// 1 square of blocks from physical image dimensions
dim3 getGridFromImage(int imageRows, int imageCols)
{
	return dim3(getGridSide(imageRows, imageCols), 
		getGridSide(imageRows, imageCols));
}

/* 
	Allow threads to malloc the memory needed for their computation 
	If this can't be done program will crash
*/
void incrementGPUHeap(size_t newHeapSize, size_t featureSize){
	cudaCheckError(cudaDeviceSetLimit(cudaLimitMallocHeapSize,  newHeapSize));
	cudaDeviceGetLimit(&newHeapSize, cudaLimitMallocHeapSize);
	cout << "\tGPU threads space: (MB) " << newHeapSize / 1024 / 1024 << endl;
	size_t free, total;
	cudaMemGetInfo(&free,&total);
	cout << "\tGPU free memory: (MB) " << (free - newHeapSize - featureSize) / 1024/1024 << endl;
}

/* 
	What to do if the default kernel configuration is not enough 
	Only for very obsolete GPUs (less than 1 GB Ram probably)
*/
void handleInsufficientMemory(){
	cerr << "FAILURE ! Gpu doesn't have enough memory \
	to hold the results and the space needed to threads" << endl;
	cerr << "Try lowering window side and/or symmetricity "<< endl;
	exit(-1);
}  

// See if the proposed number of threads will have enough memory
bool checkEnoughWorkingAreaForThreads(int numberOfPairs, int numberOfThreads,
 size_t featureSize){
	// Get GPU mem size
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	size_t gpuMemory = prop.totalGlobalMem;
	// Compute the memory needed for a single thread
	size_t workAreaSpace = numberOfPairs * 
		( 4 * sizeof(AggregatedGrayPair) + 1 * sizeof(GrayPair));
	// Multiply per threads number
	size_t totalWorkAreas = workAreaSpace * numberOfThreads;
	// add some space needed for local variables
	int FACTOR = 2; 
	totalWorkAreas *= FACTOR;

	long long int difference = gpuMemory - (featureSize + totalWorkAreas);
	if(difference <= 0){
		return false;
	}
	else{
		// Allow the GPU threads to allocate the necessary space
		incrementGPUHeap(totalWorkAreas, featureSize);
		return true;
	}
}

// will return the smallest grid that can fit into the GPU memory
dim3 getGridFromAvailableMemory(int numberOfPairs,
 size_t featureSize){

	// Get GPU mem size
	size_t freeGpuMemory, total;
	cudaMemGetInfo(&freeGpuMemory,&total);

	// Compute the memory needed for a single thread
	size_t workAreaSpace = numberOfPairs * 
		( 4 * sizeof(AggregatedGrayPair) + 1 * sizeof(GrayPair));
	// add some space needed for local variables
	int FACTOR = 2; 
	workAreaSpace *= FACTOR;

	// how many thread fit into a single block
	int threadInBlock = getCudaBlockSideX() * getCudaBlockSideX();

	size_t singleBlockMemoryOccupation = workAreaSpace * threadInBlock;
	// Even 1 block can be launched
	if(freeGpuMemory <= singleBlockMemoryOccupation){
		handleInsufficientMemory(); // exit
	}

	cout << "WARNING! Maximum available gpu memory consumed" << endl;
	// how many blocks can be launched
	int numberOfBlocks = freeGpuMemory / singleBlockMemoryOccupation;
	
	// Create 2d grid of blocks
	int gridSide = sqrt(numberOfBlocks);
	return dim3(gridSide, gridSide);
}


/* 
	Method that will generate the computing grid 
	Gpu allocable heap will be changed according to the grid individuated
	If not even 1 block can be launched the program will abort
*/
dim3 getGrid(int numberOfPairsInWindow, size_t featureSize, int imgRows, int imgCols){
 	dim3 Blocks = getBlockConfiguration();
	// Generate grid from image dimensions
	dim3 Grid = getGridFromImage(imgRows, imgCols);
	// check if there is enough space on the GPU to allocate working areas
	int numberOfBlocks = Grid.x * Grid.y;
	int numberOfThreadsPerBlock = Blocks.x * Blocks.y;
	int numberOfThreads = numberOfThreadsPerBlock * numberOfBlocks;
	if(! checkEnoughWorkingAreaForThreads(numberOfPairsInWindow, numberOfThreads, featureSize))
	{
		Grid = getGridFromAvailableMemory(numberOfPairsInWindow, featureSize);
		// Get the total number of threads and see if the gpu memory is sufficient
		numberOfBlocks = Grid.x * Grid.y;
		numberOfThreads = numberOfThreadsPerBlock * numberOfBlocks;
		checkEnoughWorkingAreaForThreads(numberOfPairsInWindow, numberOfThreads, featureSize);
	}
	printGPULaunchConfiguration(Grid, Blocks);
	return Grid;
}


/* 
	Each threads will get the memory needed for its computation 
*/
__device__ WorkArea generateThreadWorkArea(int numberOfPairs, 
	double* d_featuresList){
	// Each 1 of these data structures allow 1 thread to work
	GrayPair* d_elements;
	AggregatedGrayPair* d_summedPairs;
	AggregatedGrayPair* d_subtractedPairs;
	AggregatedGrayPair* d_xMarginalPairs;
	AggregatedGrayPair* d_yMarginalPairs;

	d_elements = (GrayPair*) malloc(sizeof(GrayPair) 
		* numberOfPairs );
	d_summedPairs = (AggregatedGrayPair*) malloc(sizeof(AggregatedGrayPair) 
		* numberOfPairs );
	d_subtractedPairs = (AggregatedGrayPair*) malloc(sizeof(AggregatedGrayPair) 
		* numberOfPairs );
	d_xMarginalPairs = (AggregatedGrayPair*) malloc(sizeof(AggregatedGrayPair) 
		* numberOfPairs);
	d_yMarginalPairs = (AggregatedGrayPair*) malloc(sizeof(AggregatedGrayPair) 
		* numberOfPairs);
	// check if allocated correctly
	WorkArea wa(numberOfPairs, d_elements, d_summedPairs,
				d_subtractedPairs, d_xMarginalPairs, d_yMarginalPairs, d_featuresList);
	return wa;
}

/*
	This kernel will iterate only when the GPU doesn't have enough memory for 
	allowing 1 thread to compute only 1 window or when 
	the image has recatungar shape
*/
__global__ void computeFeatures(unsigned int * pixels, 
	ImageData img, Window windowData, int numberOfPairsInWindow, 
	double* d_featuresList){
	// Memory location on which the thread will work
	WorkArea wa = generateThreadWorkArea(numberOfPairsInWindow, d_featuresList);
	// Get X and Y starting coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x; 
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// If 1 thread need to compute more than 1 window 
	// How many shift right for reaching the next window to compute
	int colsStride =  gridDim.x * blockDim.x; 
	// How many shift down for reaching the next window to compute
	int rowsStride =  gridDim.y * blockDim.y;
	
	// Create local window information
	Window actualWindow {windowData.side, windowData.distance,
								 windowData.directionType, windowData.symmetric};
	for(int j = x; (j + windowData.side) <= img.getColumns() ; j+= colsStride){
		for(int i = y; (i + windowData.side) <= img.getRows(); i+= rowsStride){
			// tell the window its relative offset (starting point) inside the image
			actualWindow.setSpacialOffsets(i, j);
			// Launch the computation of features on the window
			WindowFeatureComputer wfc(pixels, img, actualWindow, wa);
		}
	}
	wa.release();
}


/* Need to call after kernel invocation */
void checkKernelLaunchError(){
	cudaError_t errSync  = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess) // Detect configuration launch errors
		printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess) // Detect kernel execution errors
		printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}
