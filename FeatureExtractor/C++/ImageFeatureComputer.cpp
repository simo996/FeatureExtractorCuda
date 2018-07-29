//
// Created by simo on 25/07/18.
//

#include "ImageFeatureComputer.h"

ImageFeatureComputer::ImageFeatureComputer(const Image& img, const Window& wd)
        : image(img), windowData(wd){
    this->windowData = wd;
}

vector<WindowFeatures> ImageFeatureComputer::computeAllFeatures(){
	vector<WindowFeatures> featuresList;

	for(int i = 0; (i + windowData.dimension) < image.getRows(); i++){
		for(int j = 0; (j + windowData.dimension) < image.getColumns() ; j++){
    		// Create local window information
    		Window actualWindow {windowData.dimension, windowData.distance,
    			windowData.symmetric};
    		actualWindow.setSpacialOffsets(i,j);
    		// Launch the computation of features on the window
    		WindowFeatureComputer wfc(image, actualWindow);
    		WindowFeatures wfs = wfc.computeBundledFeatures();
    		// save results
    		featuresList.push_back(wfs);
		}
	}

	return featuresList;
}
