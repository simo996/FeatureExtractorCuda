#include <iostream>
#include <fstream>
#include <sys/stat.h>

#include "CudaFunctions.h"
#include "ImageFeatureComputer.h"


ImageFeatureComputer::ImageFeatureComputer(const ProgramArguments& progArg)
:progArg(progArg){}

/**
 * Display a set of information about the computation of the provided image
 * @param imgData
 * @param padding
 */
void ImageFeatureComputer::printInfo(const ImageData imgData, int border) {
	cout << endl << "- Input image: " << progArg.imagePath;
	cout << endl << "- Output folder: " << progArg.outputFolder;
	int pixelCount = imgData.getRows() * imgData.getColumns();
	int rows = imgData.getRows() - 2 * getAppliedBorders();
    int cols = imgData.getColumns() - 2 * getAppliedBorders();
	cout << endl << "- Rows: " << rows << " - Columns: " << cols << " - Pixel count: " << pixelCount;
	if(progArg.verbose)
		cout << endl << "- Image weight (MB): " << (pixelCount 
			* sizeof(unsigned int)) / 1024 / 1024;
	cout << endl << "- Gray Levels : " << imgData.getMaxGrayLevel();
	cout << endl << "- Distance: " << progArg.distance;
	cout << endl << "- Window side: " << progArg.windowSize;
}

/**
 * Display the memory space used while computing the problem
 * @param imgData
 * @param padding
 */
void ImageFeatureComputer::printExtimatedSizes(const ImageData& img){
    int numberOfRows = img.getRows() - progArg.windowSize + 1;
    int numberOfColumns = img.getColumns() - progArg.windowSize + 1;
    int numberOfWindows = numberOfRows * numberOfColumns;
    int supportedFeatures = Features::getSupportedFeaturesCount();

    int featureNumber = numberOfWindows * supportedFeatures;
    cout << endl << "* Size estimation * " << endl;
    cout << "\tTotal features number: " << featureNumber << endl;
    int featureSize = (((featureNumber * sizeof(double))
                        /1024)/1024);
    cout << "\tTotal features weight: " <<  featureSize << " MB" << endl;
}

/**
 * Check if all the options are coherent with the image read
 * @param progArg
 * @param img
 */
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

/**
 * Utility method
 * @return applied border to the original image read
 */
int ImageFeatureComputer::getAppliedBorders(){
    int bordersToApply = 0;
    if(progArg.borderType != 0 )
        bordersToApply = progArg.windowSize;
    return bordersToApply;
}

/**
 * This method will read the image, compute the features, re-arrange the
 * results and save them as need on the file system
 */
void ImageFeatureComputer::compute(){
	bool verbose = progArg.verbose;

	// Image from imageLoader
	Image image = ImageLoader::readImage(progArg.imagePath, progArg.borderType,
                                         getAppliedBorders(), progArg.quantitize,
                                         progArg.quantitizationMax);
	ImageData imgData(image, getAppliedBorders());
	if(verbose)
    	cout << "* Image loaded * ";
    checkOptionCompatibility(progArg, image);
    // Print computation info to cout
	printInfo(imgData, progArg.distance);
	if(verbose) {
		// Additional info on memory occupation
		printExtimatedSizes(imgData);
	}

	if(verbose)
		cout << "* COMPUTING features * " << endl;
	vector<vector<WindowFeatures>> fs= computeAllFeatures(image.getPixels().data(), imgData);
	vector<vector<FeatureValues>> formattedFeatures = getAllDirectionsAllFeatureValues(fs);
	if(verbose)
		cout << "* Features computed * " << endl;

	// Save result to file
	if(verbose)
		cout << "* Saving features to files *" << endl;
	saveFeaturesToFiles(formattedFeatures);

	// Save feature images
	if(progArg.createImages){
		if(verbose)
			cout << "* Creating feature images *" << endl;
		// Compute how many features will be used for creating the image
		int realImageRows = image.getRows() - 2 * getAppliedBorders();
        int realImageCols = image.getColumns() - 2 * getAppliedBorders();
        saveAllFeatureImages(realImageRows, realImageCols, formattedFeatures);
	}
	if(verbose)
		cout << "* DONE * " << endl;
}



/**
 * This method will re-arrange (de-linearize) all the feature values
 * computed window per window in a structure organized as features
 * values of a feature, for each direction, for each window of the image
 * @param featureValues: all the features computed, window per window
 * @param numberOfWindows: how many windows were computed
 * @param featuresCount: how many features were computed in each window
 * @return structured array (windowFeatures [] where each cell has
 * directionFeatures[] where each cell has double[] = features)
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


WorkArea generateGlobalWorkArea(int numberOfPairs, int numberOfThreads,
	double* d_featuresList){
	
	int totalNumberOfPairs = numberOfPairs * numberOfThreads;
	
	// Each 1 of these data structures allow 1 thread to work
	GrayPair* d_grayParis;
	AggregatedGrayPair* d_summedPairs;
	AggregatedGrayPair* d_subtractedPairs;
	AggregatedGrayPair* d_xMarginalPairs;
	AggregatedGrayPair* d_yMarginalPairs;

	cudaCheckError(cudaMalloc(&d_grayParis, sizeof(GrayPair) * 
		totalNumberOfPairs));
	cudaCheckError(cudaMalloc(&d_summedPairs, sizeof(AggregatedGrayPair) * 
		totalNumberOfPairs));
	cudaCheckError(cudaMalloc(&d_subtractedPairs, sizeof(AggregatedGrayPair) * 
		totalNumberOfPairs));
	cudaCheckError(cudaMalloc(&d_xMarginalPairs, sizeof(AggregatedGrayPair) * 
		totalNumberOfPairs));
	cudaCheckError(cudaMalloc(&d_yMarginalPairs, sizeof(AggregatedGrayPair) * 
		totalNumberOfPairs));

	WorkArea wa(numberOfPairs, d_grayParis, d_summedPairs,
				d_subtractedPairs, d_xMarginalPairs, d_yMarginalPairs, d_featuresList);
	return wa;
}


/**
 * This method will compute all the features for every window for the
 * number of directions provided
 * @param pixels: pixels intensities of the image provided
 * @param img: image metadata
 * @return array (1 for each window) of array (1 for each computed direction)
 * of array of doubles (1 for each feature)
 */
vector<vector<WindowFeatures>> ImageFeatureComputer::computeAllFeatures(unsigned int * pixels, const ImageData& img){
	bool verbose = progArg.verbose;
	if(verbose)
		queryGPUData();

	// Create window structure that will be given to threads
	Window windowData = Window(progArg.windowSize, progArg.distance, 
		progArg.directionType, progArg.symmetric);

  	int realImageRows = img.getRows() - 2 * getAppliedBorders();
    int realImageCols = img.getColumns() - 2 * getAppliedBorders();

	// How many windows need to be allocated
    int numberOfWindows = (realImageRows * realImageCols);
	// How many directions need to be allocated for each window
	short int numberOfDirs = 1;
	// How many feature values need to be allocated for each direction
	int featuresCount = Features::getSupportedFeaturesCount();

	// Pre-Allocate the array that will contain features
	size_t featureSize = numberOfWindows * numberOfDirs * featuresCount * sizeof(double);
	double* featuresList = (double*) malloc(featureSize);
	if(featuresList == NULL){
		cerr << "FATAL ERROR! Not enough mallocable memory on the system" << endl;
		exit(3);
	}

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
		img.getRows(), img.getColumns(), verbose);

	int numberOfThreads = Grid.x * Grid.y * Blocks.x * Blocks.y;

	// CPU pre-allocation of the device memory consumed by threads
	WorkArea globalWorkArea = generateGlobalWorkArea(numberOfPairsInWindow, 
		numberOfThreads, d_featuresList);

	// Launch the kernel
	computeFeatures<<<Grid, Blocks>>>(d_pixels, img, windowData, 
			globalWorkArea);

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
	globalWorkArea.release(); // Release device work memory
	cudaFree(d_featuresList); // release Gpu feature array
	cudaFree(d_pixels); // release Gpu image
	
	return output;
}


/**
 * This method will extract the results from each window
 * @param imageFeatures: array (1 for each window) of array (1 for each
 * computed direction) of array of doubles (1 for each feature)
 * @return array (1 for each direction) of array (1 for each feature) of all
 * the values computed of that feature
 * Es. <Entropy , (0.1, 0.2, 3, 4 , ...)>
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


/**
 * This method will save on different folders, all the features values
 * computed for each directions of the image
 * @param imageFeatures
 */
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

/**
 * This method will save into the given folder, alle the values of all
 * the features computed for 1  directions
 * @param imageDirectedFeatures: all the values computed for each feature
 * in 1 direction of the image
 * @param outputFolderPath
 */
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

/**
 * This method will save into the given folder, all the values for 1 feature
 * computed for 1 directions
 * @param imageFeatures all the feature values of 1 feature
 * @param path
 */
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

// IMAGING
/**
 * This method will produce and save all the images associated with each feature
 * for each direction
 * @param rowNumber: how many rows each image will have
 * @param colNumber: how many columns each image will have
 * @param imageFeatures
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

/**
 * This method will produce and save all the images associated with
 * each feature in 1 direction
 * @param rowNumber: how many rows each image will have
 * @param colNumber: how many columns each image will have
 * @param imageFeatures: all the values computed for each feature of the image
 * @param outputFolderPath: where to save the image
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

/**
 * This method will produce and save on the filesystem the image associated with
 * a feature in 1 direction
 * @param rowNumber: how many rows the image will have
 * @param colNumber: how many columns the image will have
 * @param featureValues: values that will be the intensities values of the
 * image
 * @param outputFilePath: where to save the image
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

