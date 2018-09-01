/*
 * ImageFeatureComputer.cpp
 *
 *  Created on: 26/ago/2018
 *      Author: simone
 */

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
	cout << "* LOADING image * " << endl;

	// Image from imageLoader
	Image image = ImageLoader::readImage(progArg.imagePath, progArg.crop);
	ImageData imgData(image);
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


int getCudaBlockSideX(){
	return 12;
}

// Square
int getCudaBlockSideY(){
	return getCudaBlockSideX();
}

dim3 getBlockConfiguration()
{
	// TODO implement GPU architecture specific dimensioning 
	int ROWS = getCudaBlockSideY();
	int COLS = getCudaBlockSideX(); 
	assert(ROWS * COLS <= 256);
	dim3 configuration(ROWS, COLS);
	return configuration;
}

int getGridSide(int imageRows, int imageCols){
	int imageSmallestSide = imageRows;
	if(imageCols < imageSmallestSide)
		imageSmallestSide = imageCols;
   
	int blockSide = getCudaBlockSideX();
	// round up division 
	int gridSide = (imageSmallestSide + blockSide -1) / blockSide;
	// Cant' exceed 65536 blocks in grid
	if(gridSide > 256){
		gridSide = 256;
	}
	return gridSide;
}

// 1 square of blocks
dim3 getGridConfiguration(int imageRows, int imageCols)
{
	return dim3(getGridSide(imageRows, imageCols), 
		getGridSide(imageRows, imageCols));
}

// Fallback Grid dimensioning when memory is not enough
dim3 getDefaultBlock(){
	return dim3(12, 12);
}

dim3 getDefaultGrid(){
	return dim3(8, 8);
}

void cudaCheckError(cudaError_t err){
	if( err != cudaSuccess ) {
		cerr << "ERROR: " << cudaGetErrorString(err) << endl;
		exit(-1);
	}
}

void incrementGPUHeap(size_t newHeapSize){
	cudaCheckError(cudaDeviceSetLimit(cudaLimitMallocHeapSize,  newHeapSize));
	cudaDeviceGetLimit(&newHeapSize, cudaLimitMallocHeapSize);
	cout << "\tWorkAreas size: (MB) " << newHeapSize/1024/1024 << endl;
	size_t free, total;
	cudaMemGetInfo(&free,&total);
	cout << "\tGPU free memory: (MB) " << free / 1024/1024 << endl;
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
		// Allow the GPU to allocate the necessary space
		incrementGPUHeap(totalWorkAreas);
		return true;
	}
}

/*	
	Print data about kernel launch configuration
*/
void printGPULaunchConfiguration(dim3 Grid, dim3 Blocks){
	cout << "\t- GPU Launch Configuration -" << endl;
	cout << "\t GRID\t rows: " << Grid.y << " x cols: " << Grid.x << endl;
	cout << "\t BLOCK\t rows: " << Blocks.y << " x cols: " << Blocks.x << endl;
}

/*
	Print data about the GPU
*/
void queryGPUData(){
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	cout << "\t- GPU DATA -" << endl;
	cout << "\tDevice name: " << prop.name << endl;
	cout << "\tNumber of multiprocessors: " << prop.multiProcessorCount << endl;
	size_t gpuMemory = prop.totalGlobalMem;
	cout << "\tTotalGlobalMemory: " << (gpuMemory/1024/2014) << " MB" << endl;
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
	This kernel will be called when the GPU doesn't have enough memory for 
	allowing 1 thread to compute only 1 window 
*/
__global__ void computeFeaturesMemoryEfficient(unsigned int * pixels, 
	ImageData img, Window windowData, int numberOfPairsInWindow, 
	double* d_featuresList){
	// Memory location on which the thread will work
	WorkArea wa = generateThreadWorkArea(numberOfPairsInWindow, d_featuresList);

	int x = blockIdx.x * blockDim.x + threadIdx.x; 
	int y = blockIdx.y * blockDim.y + threadIdx.y;
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

/* 1 thread will compute the features for 1 window */
__global__ void computeFeatures(unsigned int * pixels, 
	ImageData img, Window windowData, int numberOfPairsInWindow, 
	double* d_featuresList){
	// Memory location on which the thread will work
	WorkArea wa = generateThreadWorkArea(numberOfPairsInWindow, d_featuresList);
	int x = blockIdx.x * blockDim.x + threadIdx.x; 
	int y = blockIdx.y * blockDim.y + threadIdx.y; 
	
	// Create local window information
	Window actualWindow {windowData.side, windowData.distance,
								 windowData.directionType, windowData.symmetric};
	if((x + windowData.side) <= img.getRows()){
		if((y + windowData.side) <= img.getColumns()){
			// tell the window its relative offset (starting point) inside the image
			actualWindow.setSpacialOffsets(x, y);
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

	// Get Grid and block configuration from physical image
	dim3 Blocks = getBlockConfiguration(); 
	dim3 Grid = getGridConfiguration(img.getRows(), img.getColumns());

	// check if there is enough space on the GPU to allocate working areas
	int numberOfBlocks = Grid.x * Grid.y;
	int numberOfThreadsPerBlock = Blocks.x * Blocks.y;
	int numberOfThreads = numberOfThreadsPerBlock * numberOfBlocks;
	if(checkEnoughWorkingAreaForThreads(numberOfPairsInWindow, numberOfThreads, featureSize))
	{
		printGPULaunchConfiguration(Grid, Blocks);
		computeFeatures<<<Grid, Blocks>>>(d_pixels, img, windowData, 
			numberOfPairsInWindow, d_featuresList);
	}
	else{
		// 5k+ threads
		Grid = getDefaultGrid();
		Blocks = getDefaultBlock();
		// Get the total number of threads and see if the gpu memory is sufficient
		int numberOfBlocks = Grid.x * Grid.y;
		int numberOfThreadsPerBlock = Blocks.x * Blocks.y;
		int numberOfThreads = numberOfThreadsPerBlock * numberOfBlocks;
		checkEnoughWorkingAreaForThreads(numberOfPairsInWindow, numberOfThreads, featureSize);
		printGPULaunchConfiguration(Grid, Blocks);
		computeFeaturesMemoryEfficient<<<Grid, Blocks>>>(d_pixels, img, windowData, 
			numberOfPairsInWindow, d_featuresList);
	}
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

void ImageFeatureComputer::saveFeaturesToFiles(const vector<vector<vector<double>>>& imageFeatures){
	string foldersPath[] ={ "Values0/", "Values45/", "Values90/", "Values135/"};
	int dirType = progArg.directionType;
    int numberOfDirs = progArg.directionsNumber; // just 1 at the moment

    for (int i = 0; i < numberOfDirs; ++i) {
        // First create the the folder
        createFolder(foldersPath[i]);
        saveDirectedFeaturesToFiles(imageFeatures[i], foldersPath[dirType -1]);
    }
}

void ImageFeatureComputer::saveDirectedFeaturesToFiles(const vector<vector<double>>& imageDirectedFeatures,
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
		const int colNumber, const vector<vector<vector<double>>>& imageFeatures){
	string foldersPath[] ={ "Images0/", "Images45/", "Images90/", "Images135/"};
	int dirType = progArg.directionType;

	 // For each direction computed
    for(int i=0; i < imageFeatures.size(); i++){
        // Create the folder
        createFolder(foldersPath[i]);
        saveAllFeatureDirectedImages(rowNumber, colNumber, imageFeatures[i], foldersPath[dirType -1]);
    }}

/*
 * This method will create ALL the images associated with each feature,
 * for 1 direction evaluated.
*/
void ImageFeatureComputer::saveAllFeatureDirectedImages(const int rowNumber,
		const int colNumber, const vector<vector<double>>& imageDirectedFeatures, const string& outputFolderPath){

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
		const int colNumber, const vector<double>& featureValues,const string& filePath){
	typedef vector<WindowFeatures>::const_iterator VI;

	int imageSize = rowNumber * colNumber;

	// Check if dimensions are compatible
	if(featureValues.size() != imageSize){
		cerr << "Fatal Error! Couldn't create the image; size unexpected " << featureValues.size();
		exit(-2);
	}

	// Create a 2d matrix of double grayPairs
	Mat_<double> imageFeature = ImageLoader::createDoubleMat(rowNumber, colNumber, featureValues);
	// Transform to a format printable to file
	Mat convertedImage = ImageLoader::convertToGrayScale(imageFeature);
	ImageLoader::stretchAndSave(convertedImage, filePath);
}

