#include <iostream>
#include <fstream>
#include <sys/stat.h>

#include "CudaFunctions.h"
#include "ImageFeatureComputer.h"


ImageFeatureComputer::ImageFeatureComputer(const ProgramArguments& progArg)
:progArg(progArg){}

void ImageFeatureComputer::printInfo(const ImageData imgData, int padding) {
	cout << "- Input image: " << progArg.imagePath;
	cout << endl << "- Output folder: " << progArg.outputFolder;
	int pixelCount = imgData.getRows() * imgData.getColumns();
	int rows = imgData.getRows() - padding -1;
    int cols = imgData.getColumns() - padding -1;
	cout << endl << "- Rows: " << rows << " - Columns: " << cols << " - Pixel count: " << pixelCount;
	cout << endl << "- Gray Levels : " << imgData.getMaxGrayLevel();
	cout << endl << "- Distance: " << progArg.distance;
	cout << endl << "- Window side: " << progArg.windowSize;
}

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
	bool verbose = progArg.verbose;

	// Image from imageLoader
	Image image = ImageLoader::readImage(progArg.imagePath, progArg.crop, progArg.distance);
	ImageData imgData(image);
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
		int numberOfRows = image.getRows() - progArg.windowSize + 1;
        int numberOfColumns = image.getColumns() - progArg.windowSize + 1;
        saveAllFeatureImages(numberOfRows, numberOfColumns, formattedFeatures);

	}
	if(verbose)
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

/*
	* This method will compute all the features for every window for the
	* number of directions provided, in a window
	* By default all 4 directions are considered; order is 0->45->90->135°
*/
vector<vector<WindowFeatures>> ImageFeatureComputer::computeAllFeatures(unsigned int * pixels, const ImageData& img){
	bool verbose = progArg.verbose;
	if(verbose)
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
		img.getRows(), img.getColumns(), verbose);

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

