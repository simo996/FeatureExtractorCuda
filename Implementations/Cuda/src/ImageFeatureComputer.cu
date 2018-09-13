#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include "ImageFeatureComputer.h"


ImageFeatureComputer::ImageFeatureComputer(const ProgramArguments& progArg)
:progArg(progArg){}


void checkOptionCompatibility(ProgramArguments& progArg, const Image img){
	int imageSmallestSide = img.getRows();
	if(img.getColumns() < imageSmallestSide)
		imageSmallestSide = img.getColumns();
	if(progArg.windowSize > imageSmallestSide){
		cout << "WARNING! The window side specified with the option -w"
				"exceeds the smallest dimension (" << imageSmallestSide << ") of the image read!" << endl;
		cout << "Window side is corrected to (" << imageSmallestSide << ")" << endl;
		progArg.windowSize = imageSmallestSide;
	}

}

void ImageFeatureComputer::compute(){
	cout << endl << "* LOADING image * " << endl;

	// Image from imageLoader
	Image image = ImageLoader::readImage(progArg.imagePath, progArg.crop, progArg.quantitize,
	        progArg.quantitizationMax, progArg.distance);	ImageData imgData(image);
	cout << "* Image loaded * " << endl;
	checkOptionCompatibility(progArg, image);
	printExtimatedSizes(imgData);


	// Compute every feature
	cout << "* COMPUTING features * " << endl;
	vector<vector<WindowFeatures>> fs= computeAllFeatures(image.getPixels().data(), imgData);
	vector<vector<FeatureValues>> formattedFeatures = getAllDirectionsAllFeatureValues(fs);
	cout << "* Features computed * " << endl;

	// Save result to file
	cout << "* Saving features to files *" << endl;
	saveFeaturesToFiles(formattedFeatures);

	// Save feature images
	if(progArg.createImages){
		cout << "* Creating feature images *" << endl;
		// Compute how many features will be used for creating the image
		int numberOfRows = image.getRows() - progArg.windowSize + 1;
		int numberOfColumns = image.getColumns() - progArg.windowSize + 1;
		saveAllFeatureImages(numberOfRows, numberOfColumns, formattedFeatures);

	}
	cout << "* DONE * " << endl;
}

void checkMinimumMemoryOccupation(size_t featureSize){
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	size_t gpuMemory = prop.totalGlobalMem;
	if(featureSize > gpuMemory){
		cerr << "FAILURE ! Gpu doesn't have enough memory \
	to hold the results" << endl;
	exit(-1);
	}
}

void ImageFeatureComputer::printExtimatedSizes(const ImageData& img){
	int numberOfRows = img.getRows() - progArg.windowSize + 1;
	int numberOfColumns = img.getColumns() - progArg.windowSize + 1;
	int numberOfWindows = numberOfRows * numberOfColumns;
	int supportedFeatures = Features::getSupportedFeaturesCount();

	int featureNumber = numberOfWindows * supportedFeatures;
	cout << "\t- Size estimation - " << endl;
	cout << "\tTotal windows number: " << numberOfWindows << endl;
	cout << "\tTotal features number: " << featureNumber << endl;
	checkMinimumMemoryOccupation(featureNumber * sizeof(double));
	int featureSize = (((featureNumber * sizeof(double))
		/1024)/1024); // in MB
	cout << "\tTotal features weight: " <<  featureSize << " MB" << endl;
}



/*
 * From linear to structured array of windowsFeature each containing
 * an array of directional features each containing all the feature values
*/
vector<vector<vector<double>>> formatOutputResults(const double* featureValues,
												   const int numberOfWindows, const int featuresCount){
	// For each window, an array of directions, of features
	vector<vector<vector<double>>> output(numberOfWindows,
										  vector<vector<double>>(1, vector<double> (featuresCount)));
	// How many double values fit into a window
	int windowResultsSize = featuresCount;

	for (int k = 0; k < numberOfWindows; ++k) {
		int windowOffset = k * windowResultsSize;
		const double* windowResultsStartingPoint = featureValues + windowOffset;

		// Copy each of the values
		vector<double> singleDirectionFeatures(windowResultsStartingPoint,
				windowResultsStartingPoint + windowResultsSize);
		output[k][0] = singleDirectionFeatures;
	}

	return output;
}

/* CUDA METHODS */

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

/*
	* This method will compute all the features for every window for the
	* number of directions provided, in a window
	* By default all 4 directions are considered; order is 0->45->90->135°
*/
vector<vector<WindowFeatures>> ImageFeatureComputer::computeAllFeatures(unsigned int * pixels, const ImageData& img){
	queryGPUData();

	// Create window structure that will be given to threads
	Window windowData = Window(progArg.windowSize, progArg.distance, 
		progArg.directionType, progArg.symmetric);

	// How many windows need to be allocated
	int numberOfWindows = (img.getRows() - progArg.windowSize + 1)
						  * (img.getColumns() - progArg.windowSize + 1);
	// How many directions need to be allocated for each window
	short int numberOfDirs = 1;
	// How many feature values need to be allocated for each direction
	int featuresCount = Features::getSupportedFeaturesCount();

	// Pre-Allocate the array that will contain features
	size_t featureSize = numberOfWindows * numberOfDirs * featuresCount * sizeof(double);
	double* featuresList = (double*) malloc(featureSize);

	// Allocate GPU space to contain results
	double* d_featuresList;
	cudaCheckError(cudaMalloc(&d_featuresList, featureSize));

	// 	Compute how many elements will be stored in each thread working memory
	int extimatedWindowRows = windowData.side; // 0° has all rows
	int extimateWindowCols = windowData.side - (windowData.distance * 1); // at least 1 column is absent
	int numberOfPairsInWindow = extimatedWindowRows * extimateWindowCols;
	if(windowData.symmetric)
		numberOfPairsInWindow *= 2;

	// COPY the image pixels to the GPU
	unsigned int * d_pixels;
	cudaCheckError(cudaMalloc(&d_pixels, sizeof(unsigned int) * img.getRows() * img.getColumns()));
	cudaCheckError(cudaMemcpy(d_pixels, pixels,
			sizeof(unsigned int) * img.getRows() * img.getColumns(),
			cudaMemcpyHostToDevice));

	// try to squiize more performance
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	// Get Grid and block configuration
	dim3 Blocks = getBlockConfiguration(); 
	dim3 Grid = getGrid(numberOfPairsInWindow, featureSize, 
		img.getRows(), img.getColumns());

	// Launch GPU computation
	computeFeatures<<<Grid, Blocks>>>(d_pixels, img, windowData, 
			numberOfPairsInWindow, d_featuresList);
	// Check if everything is ok
	checkKernelLaunchError();

	// Copy back results from GPU
	cudaCheckError(cudaMemcpy(featuresList, d_featuresList,
			featureSize,
			cudaMemcpyDeviceToHost));

	// Give the data structure
	// Windows[] with Directions[] with feature values
	vector<vector<vector<double>>> output =
			formatOutputResults(featuresList, numberOfWindows, featuresCount);

	free(featuresList); // release Cpu feature array
	cudaFree(d_featuresList); // release Gpu feature array
	cudaFree(d_pixels); // release Gpu image
	
	return output;
}


/*
 * This method will generate a vector of vectors (1 for each direction)
  of features names and all their values found in the image
 * Es. <Entropy , (0.1, 0.2, 3, 4 , ...)>
 * Es. <IMOC, (-1,-2,0)>
 */
vector<vector<vector<double>>> ImageFeatureComputer::getAllDirectionsAllFeatureValues(const vector<vector<vector<double>>>& imageFeatures){
	vector<FeatureNames> supportedFeatures = Features::getAllSupportedFeatures();
	// Direzioni[] aventi Features[] aventi double[]
	vector<vector<vector<double>>> output(1);

	// for each computed direction
	// 1 external vector cell for each of the 18 features
	// each cell has all the values of that feature
	vector<vector<double>> featuresInDirection(supportedFeatures.size());

	// for each computed window
	for (int i = 0; i < imageFeatures.size() ; ++i) {
		// for each supported feature
		for (int k = 0; k < supportedFeatures.size(); ++k) {
			FeatureNames actualFeature = supportedFeatures[k];
			// Push the value found in the output list for that direction
			featuresInDirection[actualFeature].push_back(imageFeatures.at(i).at(0).at(actualFeature));
		}

	}
	output[0] = featuresInDirection;

	return output;
}

/* Support code for putting the results in the right output folder */
void createFolder(string folderPath){
    if (mkdir(folderPath.c_str(), 0777) == -1) {
        if (errno == EEXIST) {
            // alredy exists
        } else {
            // something else
            cerr << "cannot create save folder: " << folderPath << endl
                 << "error:" << strerror(errno) << endl;
        }
    }
}


void ImageFeatureComputer::saveFeaturesToFiles(const vector<vector<FeatureValues>>& imageFeatures){
    int dirType = progArg.directionType;

    string outFolder = progArg.outputFolder;
    Utils::createFolder(outFolder);
    string foldersPath[] ={ "/Values0/", "/Values45/", "/Values90/", "/Values135/"};

    // First create the the folder
    string outputDirectionPath = outFolder + foldersPath[dirType -1];
	Utils::createFolder(outputDirectionPath);
    saveDirectedFeaturesToFiles(imageFeatures[0], outputDirectionPath);
}

void ImageFeatureComputer::saveDirectedFeaturesToFiles(const vector<FeatureValues>& imageDirectedFeatures,
		const string& outputFolderPath){
	vector<string> fileDestinations = Features::getAllFeaturesFileNames();

	// for each feature
	for(int i = 0; i < imageDirectedFeatures.size(); i++) {
		string newFileName(outputFolderPath); // create the right file path
		pair<FeatureNames , FeatureValues> featurePair = make_pair((FeatureNames) i, imageDirectedFeatures[i]);
		saveFeatureToFile(featurePair, newFileName.append(fileDestinations[i]));
	}
}

void ImageFeatureComputer::saveFeatureToFile(const pair<FeatureNames, vector<double>>& featurePair, string filePath){
	// Open the file
	ofstream file;
	file.open(filePath.append(".txt"));
	if(file.is_open()){
		for(int i = 0; i < featurePair.second.size(); i++){
			file << featurePair.second[i] << ",";
		}
		file.close();
	} else{
		cerr << "Couldn't save the feature values to file" << endl;
	}

}

/*
 * This method will create ALL the images associated with each feature,
 * for ALL the directions evaluated.
*/
void ImageFeatureComputer::saveAllFeatureImages(const int rowNumber,
		const int colNumber, const vector<vector<FeatureValues>>& imageFeatures){
    int dirType = progArg.directionType;

    string outFolder = progArg.outputFolder;
    string foldersPath[] ={ "/Images0/", "/Images45/", "/Images90/", "/Images135/"};
    string outputDirectionPath = outFolder + foldersPath[dirType -1];
	Utils::createFolder(outputDirectionPath);
    // For each direction computed
    saveAllFeatureDirectedImages(rowNumber, colNumber, imageFeatures[0],
                outputDirectionPath);
}

/*
 * This method will create ALL the images associated with each feature,
 * for 1 direction evaluated.
*/
void ImageFeatureComputer::saveAllFeatureDirectedImages(const int rowNumber,
		const int colNumber, const vector<FeatureValues>& imageDirectedFeatures, const string& outputFolderPath){

	vector<string> fileDestinations = Features::getAllFeaturesFileNames();

	// For each feature
	for(int i = 0; i < imageDirectedFeatures.size(); i++) {
		string newFileName(outputFolderPath);
		saveFeatureImage(rowNumber, colNumber, imageDirectedFeatures[i], newFileName.append(fileDestinations[i]));
	}
}

/*
 * This method will create an image associated with a feature,
 * for a single side evaluated;
*/
void ImageFeatureComputer::saveFeatureImage(const int rowNumber,
		const int colNumber, const FeatureValues& featureValues,const string& filePath){
	typedef vector<WindowFeatures>::const_iterator VI;

	int imageSize = rowNumber * colNumber;

	// Check if dimensions are compatible
	if(featureValues.size() != imageSize){
		cerr << "Fatal Error! Couldn't create the image; size unexpected " << featureValues.size();
		exit(-2);
	}

	// Create a 2d matrix of double grayPairs
	Mat_<double> imageFeature = ImageLoader::createDoubleMat(rowNumber, colNumber, featureValues);
    ImageLoader::saveImage(imageFeature, filePath);
}

