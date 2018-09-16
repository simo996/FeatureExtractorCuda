//
// Created by simo on 16/07/18.
//

#include <iostream>
#include "WindowFeatureComputer.h"


WindowFeatureComputer::WindowFeatureComputer(const Image& img, const Window& wd)
		: image(img), windowData(wd){
}

/*
	This method will compute all the features for all numberOfDirections directions
 	provided by a parameter to the program ; the order is 0,45,90,135° ;
 	By default all 4 directions are evaluated
*/
vector<vector<double>> WindowFeatureComputer::computeWindowFeatures() {
	vector<vector<double>> featureList(1);

	// Get shift vector for each direction of interest
	Direction actualDir = Direction(windowData.directionType);
	FeatureComputer fc(image, actualDir.shiftRows, actualDir.shiftColumns,
						   windowData);
	vector<double> computedFeatures = fc.computeFeatures();
	featureList[0] =  computedFeatures;
	return featureList;
}
