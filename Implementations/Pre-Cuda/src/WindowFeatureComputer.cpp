//
// Created by simo on 16/07/18.
//

#include "WindowFeatureComputer.h"

WindowFeatureComputer::WindowFeatureComputer(unsigned int * pxls,
		const ImageData& img, const Window& wd, WorkArea& wa): pixels(pxls),
		image(img), windowData(wd), workArea(wa){
	computeWindowFeatures();
}

/*
	This method will compute all the features for all numberOfDirections directions
 	provided by a parameter to the program ; the order is 0,45,90,135° ;
 	By default all 4 directions are evaluated
*/
void WindowFeatureComputer::computeWindowFeatures() {
	for(int i = 0; i < windowData.numberOfDirections; i++)
	{
		// Get shift vector for each direction of interest
		Direction actualDir = Direction(i);
		// create the autonomous thread of computation
		FeatureComputer fc(pixels, image, actualDir.shiftRows, actualDir.shiftColumns,
						   windowData, workArea, i);
	}
}