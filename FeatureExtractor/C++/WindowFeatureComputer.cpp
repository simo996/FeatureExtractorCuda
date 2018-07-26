//
// Created by simo on 16/07/18.
//

#include <iostream>
#include "WindowFeatureComputer.h"
#include "FeatureComputer.h"

WindowFeatureComputer::WindowFeatureComputer(const vector<int>& inputPixels, const int maxGrayLevel, const Window wd)
		:windowData(wd){
	this->maxGrayLevel = maxGrayLevel;
	this->inputPixels = inputPixels;
}

typedef struct Direction{
	string label;
	int shiftRows;
	int shiftColumns;
} Direction;

vector<Direction> getAllDirections(){
	Direction d0{"Direction 0°", 0, 1};
	Direction d45{"Direction 45°", -1, 1};
	Direction d90{"Direction 90°", -1, 0};
	Direction d135{"Direction 135°", -1, -1};

	vector<Direction> out = {d0, d45, d90, d135};
	return out;
}

// TODO think about keeping this function or using the bundled version
/*
	This method will compute the features for all 4 supported directions
*/
vector<map<FeatureNames, double>> WindowFeatureComputer::computeFeatures(){
    vector<map<FeatureNames, double>> featureList(4);

   	vector<Direction> allDirections = getAllDirections();

    for(int i = 0; i < 4; i++)
    {
    	Direction actualDir = allDirections[i];
  		FeatureComputer fc(inputPixels, maxGrayLevel, actualDir.shiftRows,
  			actualDir.shiftColumns, windowData);
  		map<FeatureNames, double> computedFeatures = fc.computeFeatures();
  		featureList.at(i) = computedFeatures;
    }
	
    return featureList;
}

/*
	This method will compute the features for all 4 supported directions
*/
vector<FeatureBundle> WindowFeatureComputer::computeBundledFeatures(){
	vector<FeatureBundle> featureList(4);

	vector<Direction> allDirections = getAllDirections();
	for(int i = 0; i < 4; i++)
	{
		Direction actualDir = allDirections[i];
		FeatureComputer fc(inputPixels, maxGrayLevel, actualDir.shiftRows,
						   actualDir.shiftColumns, windowData);
		map<FeatureNames, double> computedFeatures = fc.computeFeatures();
		featureList.at(i) = { actualDir.label, computedFeatures};
	}

	return featureList;
}

// TODO think about keeping this function or using the bundled version
/*
	This method will print the features for all 4 supported directions with
	an explanatory label
*/
void WindowFeatureComputer::printSeparatedFeatures(vector<map<FeatureNames, double>> featureList) const{
	vector<Direction> allDirections = getAllDirections();

	for(int i = 0; i < 4; i++) {
		Direction actualDir = allDirections[i];
		cout << "\n\t** " << actualDir.label << " **" <<endl;
		FeatureComputer::printFeatures(featureList[i]);
		cout << endl;
	}
}

/*
	This method will print the features for all 4 supported directions with
	an explanatory label
*/
void WindowFeatureComputer::printBundledFeatures(vector<FeatureBundle> featureList) const{
	for(int i = 0; i < featureList.size(); i++) {
		cout << "\n\t** " << featureList.at(i).directionLabel << " **" <<endl;
		FeatureComputer::printFeatures(featureList.at(i).features);
		cout << endl;
	}

}
