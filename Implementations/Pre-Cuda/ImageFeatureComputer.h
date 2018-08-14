/*
	Created by simo on 25/07/18.
    * RESPONSABILITA CLASSE: Scorrere l'immagine creando tutte le possibili finestre sovrapposte
*/

#ifndef FEATUREEXTRACTOR_IMAGEFEATURECOMPUTER_H
#define FEATUREEXTRACTOR_IMAGEFEATURECOMPUTER_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include "ImageLoader.h"
#include "WindowFeatureComputer.h"

using namespace cv;

struct ProgramArguments{
	// TODO document intensively
	short int windowSize;
	bool crop;
	bool symmetric;
	short int distance;
	short int numberOfDirections;
	bool createImages;
	short int chosenDevice; // 0 = gpu, 1=cpu, 'a'= auto
	string imagePath;

	ProgramArguments(short int windowSize = 4, bool crop = false, bool symmetric = false,
					 short int distance = 1, short int numberOfDirections = 4,
					 bool createImages = false, short int chosenDevice = 0)
			: windowSize(windowSize), crop(crop), symmetric(symmetric), distance(distance),
			  numberOfDirections(numberOfDirections),
			  createImages(createImages), chosenDevice(chosenDevice){}
};

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

	// This method will print all the feature names and all their values computed in the image
	void printAllDirectionsAllFeatureValues(const vector<vector<FeatureValues>>& featureList);
};


#endif //FEATUREEXTRACTOR_IMAGEFEATURECOMPUTER_H
