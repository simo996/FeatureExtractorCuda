/*
 * ImageFeatureComputer.h
 *
 *  Created on: 26/ago/2018
 *      Author: simone
 */

#ifndef IMAGEFEATURECOMPUTER_H_
#define IMAGEFEATURECOMPUTER_H_

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include "ImageLoader.h"
#include "ProgramArguments.h"
#include "WindowFeatureComputer.h"

typedef vector<vector<double>> WindowFeatures;
typedef vector<double> FeatureValues;

using namespace cv;

class ImageFeatureComputer {
public:
	ImageFeatureComputer(const ProgramArguments& progArg);

	void compute();
    vector<WindowFeatures> computeAllFeatures(unsigned int * pixels, const ImageData& img);

    // EXTRAPOLATING RESULTS
	// This method will get all the feature names and all their values computed in the image
	vector<vector<FeatureValues>> getAllDirectionsAllFeatureValues(const vector<WindowFeatures>& imageFeatures);

	// SAVING RESULTS ON FILES
	/* This method will save on different folders, the features computed for the distinct directions */
	void saveFeaturesToFiles(const vector<vector<FeatureValues>>& imageFeatures);

    // IMAGING
    // This methow will produce and save all the images associated with each feature for each direction
    void saveAllFeatureImages(int rowNumber,  int colNumber, const vector<vector<FeatureValues>>& imageFeatures);


private:
	ProgramArguments progArg;

	// SUPPORT FILESAVE methods
	/* This method will save into the given folder, the features computed for the that directions */
	void saveDirectedFeaturesToFiles(const vector<FeatureValues>& imageDirectedFeatures,
			const string& outputFolderPath);
	/* This method will save into the given folder, 1 feature computed for the that directions */
	void saveFeatureToFile(const pair<FeatureNames, vector<double>>& imageFeatures, const string path);

	// SUPPORT IMAGING methods
	// This methow will produce and save all the images associated with each feature in 1 direction
	void saveAllFeatureDirectedImages(int rowNumber,  int colNumber,
			const vector<vector<double>> &imageFeatures, const string& outputFolderPath);
	// This method will produce and save on the filesystem the image associated with a feature in 1 direction
	void saveFeatureImage(int rowNumber,  int colNumber,
			const FeatureValues& featureValues, const string& outputFilePath);

	// DEBUG info
	void printExtimatedSizes(const ImageData& img);

};



#endif /* IMAGEFEATURECOMPUTER_H_ */
