//
// Created by simo on 25/07/18.
//

#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include "ImageFeatureComputer.h"

ImageFeatureComputer::ImageFeatureComputer(const Image& img, const Window& wd)
        : image(img), windowData(wd){
}

/*
 * This method will compute all the features for every window for the
 * number of directions provided, in a window
 * By default all 4 directions are considered; order is 0->45->90->135°
 */
vector<WindowFeatures> ImageFeatureComputer::computeAllFeatures(const int numberOfDirections){
	this->numberOfDirections = numberOfDirections;
	vector<WindowFeatures> featuresList;

	for(int i = 0; (i + windowData.side) <= image.getRows(); i++){
		for(int j = 0; (j + windowData.side) <= image.getColumns() ; j++){
			// Create local window information
			Window actualWindow {windowData.side, windowData.distance,
								 windowData.symmetric};
			actualWindow.setSpacialOffsets(i,j);
			// Launch the computation of features on the window
			WindowFeatureComputer wfc(image, actualWindow);
			WindowFeatures wfs = wfc.computeWindowFeatures(numberOfDirections);
			// save results
			featuresList.push_back(wfs);
		}
	}

	return featuresList;
}



/*
 * This method will generate a vector of maps (1 for each direction) of features names and all their values found in the image
 * Es. <Entropy , (0.1, 0.2, 3, 4 , ...)>
 * Es. <IMOC, (-1,-2,0)>
 */
vector<map<FeatureNames, vector<double>>> ImageFeatureComputer::getAllDirectionsAllFeatureValues(const vector<WindowFeatures>& imageFeatures){
	vector<FeatureNames> supportedFeatures = Features::getAllSupportedFeatures();
	vector<map<FeatureNames, vector<double>>> output(numberOfDirections);

	// for each computed direction
	for (int j = 0; j < numberOfDirections; ++j) {
		map<FeatureNames, vector<double>> featuresInDirection;
		// for each computed window
		for (int i = 0; i < imageFeatures.size() ; ++i) {

			// for each supported feature
			for (int k = 0; k < supportedFeatures.size(); ++k) {
				FeatureNames actualFeature = supportedFeatures[k];
				// Push the value found in the output list for that direction
				featuresInDirection[actualFeature].push_back(imageFeatures.at(i).at(j).at(actualFeature));
			}
		}
		output[j]=featuresInDirection;
	}

	return output;
}

void ImageFeatureComputer::saveFeaturesToFiles(const vector<map<FeatureNames, vector<double>>>& imageFeatures){
	string foldersPath[] ={ "Values0/", "Values45/", "Values90/", "Values135/"};
	for (int i = 0; i < numberOfDirections; ++i) {
		saveDirectedFeaturesToFiles(imageFeatures[i], foldersPath[i]);
	}
}

void ImageFeatureComputer::saveDirectedFeaturesToFiles(const map<FeatureNames, vector<double>>& imageDirectedFeatures, const string outputFolderPath){
	typedef map<FeatureNames, vector<double>>::const_iterator MI;
	vector<string> fileDestinations = Features::getAllFeaturesFileNames();

	int i = 0;
	for(MI featurePairs = imageDirectedFeatures.begin(); featurePairs != imageDirectedFeatures.end(); featurePairs++){
		if (mkdir(outputFolderPath.c_str(), 0777) == -1) {
			if (errno == EEXIST) {
				// alredy exists
			} else {
				// something else
				cout << "cannot create save folder;  error:" << strerror(errno) << endl;
			}
		}
		string newFileName(outputFolderPath);
		saveFeatureToFile(*featurePairs, newFileName.append(fileDestinations[i]));
		i++;
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
 * This method will print the given vector of maps (1 for each direction)
 * of features names and all their values found in the image
 * Es. <Entropy , (0.1, 0.2, 3, 4 , ...)>
 * Es. <IMOC, (-1,-2,0)>
 */
void ImageFeatureComputer::printAllDirectionsAllFeatureValues(const vector<map<FeatureNames, vector<double>>>& featureList)
{
	typedef map<FeatureNames, vector<double>>::const_iterator MI;
	typedef vector<double>::const_iterator VI;
	// For each direction
	for (int i = 0; i < numberOfDirections; ++i) {
		Direction::printDirectionLabel(i);
		// for each computed feature
		for (MI mappedFeature = featureList[i].begin(); mappedFeature != featureList[i].end(); ++mappedFeature) {
			Features::printFeatureName(mappedFeature->first); // print the feature label
			// Print all values
			for(VI featureValue = mappedFeature->second.begin(); featureValue != mappedFeature->second.end(); featureValue++){
				cout << *featureValue << " ";
			}
			cout << endl;
		}
	}
}

/*
 * This method will create ALL the images associated with each feature,
 * for ALL the side evaluated.
*/
void ImageFeatureComputer::saveAllFeatureImages(const vector<map<FeatureNames, vector<double>>> &imageFeatures){
	string foldersPath[] ={ "Images0/", "Images45/", "Images90/", "Images135/"};

	// For each direction
	for(int i=0; i < imageFeatures.size(); i++){
		saveAllFeatureDirectedImages(imageFeatures[i], foldersPath[i]);
	}
}

/*
 * This method will create ALL the images associated with each feature,
 * for 1 side evaluated.
*/
void ImageFeatureComputer::saveAllFeatureDirectedImages(const map<FeatureNames, vector<double>> &imageDirectedFeatures,
														string outputFolderPath){
	typedef map<FeatureNames, vector<double>>::const_iterator MI;

	vector<string> fileDestinations = Features::getAllFeaturesFileNames();
	int i = 0;

	for(MI feature = imageDirectedFeatures.begin(); feature != imageDirectedFeatures.end(); feature++){
		saveFeatureImage(imageDirectedFeatures, feature->first,
						 outputFolderPath.append(fileDestinations[i]));
		i++;
	}
}

void saveImageToFile(const cv::Mat& img, const string fileName){
	try {
		imwrite(fileName, img);
	}catch (exception& e){
		cout << e.what() << '\n';
		cerr << "Fatal Error! Couldn't save the image";
		exit(-3);
	}
}

/*
 * This method will create an image associated with a feature,
 * for a single side evaluated;
*/
void ImageFeatureComputer::saveFeatureImage(
		const map<FeatureNames, vector<double>> &imageDirectedFeatures, FeatureNames fname, string filePath){
	typedef vector<WindowFeatures>::const_iterator VI;

	// Compute how many features will be used for creating the image
	int numberOfRows = image.getRows() / windowData.distance;
	int numberOfColumns = image.getColumns() / windowData.distance;
	int imageSize = numberOfRows * numberOfColumns;

	// Check if dimensions are compatible
	if(imageDirectedFeatures.at(fname).size() != imageSize){
		cerr << "Fatal Error! Couldn't create the image";
		exit(-2);
	}

	// Create a 2d matrix of double elements
	cv::Mat imageFeature = cv::Mat(numberOfRows, numberOfColumns, CV_64F);
	// CHECK se va a capo da solo e non fa tutto su una riga
	// Copy the values into the image
	memcpy(imageFeature.data, imageDirectedFeatures.at(fname).data(), imageSize * sizeof(double));

	//cv::imshow
	saveImageToFile(imageFeature, filePath);
}



// TODO this is a debugging method
/*
 * This method will print ALL 18 the features, for all the windows of the image,
 * in all 4 directions
 */
void ImageFeatureComputer::printImageAllDirectionsAllFeatures(const vector<WindowFeatures>& imageFeatureList){
	typedef vector<WindowFeatures>::const_iterator VI;

	for (VI windowElement = imageFeatureList.begin(); windowElement!= imageFeatureList.end() ; windowElement++)
	{
		WindowFeatureComputer::printAllDirectionsAllFeatures(*windowElement);
	}
}



