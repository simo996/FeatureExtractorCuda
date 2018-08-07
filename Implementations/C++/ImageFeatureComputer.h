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
    vector<WindowFeatures> computeAllFeatures(const Image& img);

    // EXTRAPOLATING RESULTS
	// This method will get all the feature names and all their values computed in the image
	vector<map<FeatureNames, vector<double>>> getAllDirectionsAllFeatureValues(const vector<WindowFeatures>& imageFeatures);
	// This method will print all the feature names and all their values computed in the image
	void printAllDirectionsAllFeatureValues(const vector<map<FeatureNames, vector<double>>>& featureList);

	// SAVING RESULTS ON FILES
	/* This method will save on different folders, the features computed for the distinct directions */
	void saveFeaturesToFiles(const vector<map<FeatureNames, vector<double>>>& imageFeatures);

    // IMAGING
    // This methow will produce and save all the images associated with each feature for each direction
    void saveAllFeatureImages(int rowNumber,  int colNumber,
    		const vector<map<FeatureNames, vector<double>>> &imageFeatures);

	// DEBUG, not really useful
	// Method will print, for each direction, for each window, all the features
	static void printImageAllDirectionsAllFeatures(const vector<WindowFeatures> &imageFeatureList);
private:
	ProgramArguments progArg;

	// SUPPORT FILESAVE methods
	/* This method will save into the given folder, the features computed for the that directions */
	void saveDirectedFeaturesToFiles(const map<FeatureNames, vector<double>>& imageFeatures, const string path);
	/* This method will save into the given folder, 1 feature computed for the that directions */
	void saveFeatureToFile(const pair<FeatureNames, vector<double>>& imageFeatures, const string path);

	// SUPPORT IMAGING methods
	// This method will produce and save on the filesystem the image associated with a feature in 1 direction
	void saveFeatureImage(int rowNumber,  int colNumber,
			const map<FeatureNames, vector<double>> &imageDirectedFeatures, FeatureNames fname,
						  string outputFilePath);
	// This methow will produce and save all the images associated with each feature in 1 direction
	void saveAllFeatureDirectedImages(int rowNumber,  int colNumber,
			const map<FeatureNames, vector<double>> &imageFeatures, const string outputFolderPath);

};


#endif //FEATUREEXTRACTOR_IMAGEFEATURECOMPUTER_H
