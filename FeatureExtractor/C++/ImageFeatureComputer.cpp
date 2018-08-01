//
// Created by simo on 25/07/18.
//

#include <iostream>
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

	for(int i = 0; (i + windowData.dimension) <= image.getRows(); i++){
		for(int j = 0; (j + windowData.dimension) <= image.getColumns() ; j++){
			// Create local window information
			Window actualWindow {windowData.dimension, windowData.distance,
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
 * This method will generate a vector of maps of features names and all their values found in the image
 * The vector will have an entry for each computed direction
 * Es. <Entropy , (0.1, 0.2, 3, 4 , ...)>
 * Es. <IMOC, (-1,-2,0)>
 */
vector<map<FeatureNames, vector<double>>> ImageFeatureComputer::getAllDirectionsAllFeatureValues(const vector<WindowFeatures>& imageFeatures){
	vector<FeatureNames> supportedFeatures = Features::getAllSupportedFeatures();
	vector<map<FeatureNames, vector<double>>> output(numberOfDirections);
	// Desume the number of computed direction by looking at how many elements are stored per window

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
 * This method will print ALL the features, for all the windows of the image,
 * in all 4 directions
 */
void ImageFeatureComputer::printImageAllDirectionsAllFeatures(const vector<WindowFeatures>& imageFeatureList){
    typedef vector<WindowFeatures>::const_iterator VI;

    for (VI windowElement = imageFeatureList.begin(); windowElement!= imageFeatureList.end() ; windowElement++)
    {
        WindowFeatureComputer::printAllDirectionsAllFeatures(*windowElement);
    }
}


/*
 * This method will print 1 feature, for all the windows of the image,
 * in all 4 directions
 */
void ImageFeatureComputer::printImageAllDirectionsSingleFeature(const vector<WindowFeatures>& imageFeatureList,
																FeatureNames fname){
	typedef vector<WindowFeatures>::const_iterator VI;
    // For each computed window
	for (VI windowElement = imageFeatureList.begin(); windowElement!= imageFeatureList.end() ; windowElement++)
	{
		WindowFeatureComputer::printAllDirectionsSingleFeature(*windowElement, fname);
	}
}

/*
 * This method will create an image associated with a feature,
 * for each dimension evaluated; vector[0] = image of 0°, etc.
*/
vector<cv::Mat> ImageFeatureComputer::generateFeatureImage(const vector<WindowFeatures> imageFeatures, FeatureNames fname){
	typedef vector<WindowFeatures>::const_iterator VI;

	int numberOfRows = image.getRows() / windowData.distance;
	int numberOfColumns = image.getColumns() / windowData.distance;

	for(int i = 0; i <imageFeatures.size(); i++){

	}
}


/*
 * This method will print ALL feature, for all the windows of the image,
 * in 1 direction: 0°
 */
void ImageFeatureComputer::printImageSingleDirectionsAllFeatures(const vector<WindowFeatures>& imageFeatureList){
	typedef vector<WindowFeatures>::const_iterator VI;

	for (VI windowElement = imageFeatureList.begin(); windowElement!= imageFeatureList.end() ; windowElement++)
	{
		//WindowFeatureComputer::printSingleDirectionAllFeatures(windowElement[0][0]) ;
	}
}

/*
 * This method will print 1 feature, for all the windows of the image,
 * in 1 direction: 0°
 */
void ImageFeatureComputer::printImageSingleDirectionsSingleFeature(const vector<WindowFeatures>& imageFeatureList, FeatureNames fname){
	typedef vector<WindowFeatures>::const_iterator VI;

	for (VI windowElement = imageFeatureList.begin(); windowElement!= imageFeatureList.end() ; windowElement++)
	{
		WindowFeatureComputer::printSingleDirectionSingleFeature(windowElement[0][0], fname);
	}
}