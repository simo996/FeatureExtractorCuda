//
// Created by simo on 16/07/18.
//

#include <iostream>
#include "WindowFeatureComputer.h"

vector<Direction> getAllDirections(){
	Direction d0{"Direction 0°", 0, 1};
	Direction d45{"Direction 45°", -1, 1};
	Direction d90{"Direction 90°", -1, 0};
	Direction d135{"Direction 135°", -1, -1};

	vector<Direction> out = {d0, d45, d90, d135};
	return out;
}

WindowFeatureComputer::WindowFeatureComputer(const Image& img, const Window& wd)
		: image(img), windowData(wd){
}

/*
	This method will compute all the features for all numberOfDirections directions
 	provided by a parameter to the program ; the order is 0,45,90,135° ;
 	By default all 4 directions are evaluated
*/
vector<FeatureBundle> WindowFeatureComputer::computeWindowFeatures(const int numberOfDirections) {
	vector<FeatureBundle> featureList(numberOfDirections);

	vector<Direction> allDirections = getAllDirections();
	for(int i = 0; i < numberOfDirections; i++)
	{
		Direction actualDir = allDirections[i];
		FeatureComputer fc(image, actualDir.shiftRows, actualDir.shiftColumns,
						   windowData);
		map<FeatureNames, double> computedFeatures = fc.computeDirectionalFeatures();
		featureList.at(i) = { actualDir.label, computedFeatures};
	}

	return featureList;
}

/*
	This method will print all the features for all 4 supported directions
*/
void WindowFeatureComputer::printAllDirectionsFeatures(const WindowFeatures &featureList){
	for(int i = 0; i < featureList.size(); i++) {
		printSingleDirectionAllFeatures(featureList[i]);
	}
}

/*
	This method will print the features for 1 supported direction with explanatory label
*/
void WindowFeatureComputer::printSingleDirectionAllFeatures(const FeatureBundle& featureList){
	cout << "\n\t** " << featureList.directionLabel << " **" <<endl;
	FeatureComputer::printAllFeatures(featureList.features);
	cout << endl;
}

/*
	This method will print 1 feature for all 4 supported direction with their label
*/
void WindowFeatureComputer::printAllDirectionsSingleFeature(const WindowFeatures &featureList, FeatureNames fname){
	for(int i = 0; i < featureList.size(); i++) {
		printSingleDirectionSingleFeature(featureList[i], fname);
	}
}

/*
	This method will print 1 feature for 1 supported direction with direction's label
*/
void WindowFeatureComputer::printSingleDirectionSingleFeature(const FeatureBundle& featureList, const FeatureNames fname){
	cout << "\n\t** " << featureList.directionLabel << " **" <<endl;
	FeatureComputer::printFeature(featureList.features, fname);
	cout << endl;
}