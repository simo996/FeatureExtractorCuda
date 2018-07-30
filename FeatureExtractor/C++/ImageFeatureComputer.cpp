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
 * 4 directions present in a window
 */
vector<WindowFeatures> ImageFeatureComputer::computeAllFeatures(){
	vector<WindowFeatures> featuresList;

	for(int i = 0; (i + windowData.dimension) <= image.getRows(); i++){
		for(int j = 0; (j + windowData.dimension) <= image.getColumns() ; j++){
    		// Create local window information
    		Window actualWindow {windowData.dimension, windowData.distance,
    			windowData.symmetric};
    		actualWindow.setSpacialOffsets(i,j);
    		// Launch the computation of features on the window
    		WindowFeatureComputer wfc(image, actualWindow);
    		WindowFeatures wfs = wfc.computeWindowFeatures();
    		// save results
    		featuresList.push_back(wfs);
		}
	}

	return featuresList;
}

/*
 * This method will print ALL the features, for all the windows of the image,
 * in all 4 directions
 */
void ImageFeatureComputer::printImageAllFeatures(const vector<WindowFeatures>& imageFeatureList){

    typedef vector<WindowFeatures>::const_iterator VI;
    for (VI windowElement = imageFeatureList.begin(); windowElement!= imageFeatureList.end() ; windowElement++)
    {
    	WindowFeatureComputer::printAllDirectionsFeatures(*windowElement);
    }
}


/*
 * This method will print 1 feature, for all the windows of the image,
 * in all 4 directions
 */
void ImageFeatureComputer::printImageSingleFeature(const vector<WindowFeatures>& imageFeatureList, FeatureNames fname){
	typedef vector<WindowFeatures>::const_iterator VI;
	for (VI windowElement = imageFeatureList.begin(); windowElement!= imageFeatureList.end() ; windowElement++)
	{
		WindowFeatureComputer::printAllDirectionsSingleFeature(*windowElement, fname);
	}
}