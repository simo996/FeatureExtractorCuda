//
// Created by simo on 25/07/18.
//

#include "ImageFeatureComputer.h"

ImageFeatureComputer::ImageFeatureComputer(const vector<int>& imagePixels, 
	const ImageData imgd, const WindowData wd){
    this->imagePixels = imagePixels;
    this->imgData = imgd;
    this->windowData = wd;
}

vector<WindowFeatures> ImageFeatureComputer::computeAllFeatures(){
	vector<WindowFeatures> imageFeatures;

	for(int i = 0; (i + windowData.dimension) < imgData.rows; i++){
		for(int j = 0; (j + windowData.dimension) < imgData.rows ; j++){
    		// Create local window information
    		WindowData actualWindow {windowData.dimension, windowData.distance,
    			windowData.symmetric, i, j};
    		// Launch the computation of features on the window
    		WindowFeatureComputer wfc(windowPixel, windowData.dimension, actualWindow);
    		WindowFeatures wfs = wfc.computeBundledFeatures();
    		// save results
    		imageFeatures.push_back(wfs);
		}
	}

	return imageFeatures;
}
