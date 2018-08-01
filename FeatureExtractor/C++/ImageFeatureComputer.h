/*
	Created by simo on 25/07/18.
    * RESPONSABILITA CLASSE: Scorrere l'immagine creando tutte le possibili finestre sovrapposte
*/

#ifndef FEATUREEXTRACTOR_IMAGEFEATURECOMPUTER_H
#define FEATUREEXTRACTOR_IMAGEFEATURECOMPUTER_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include "WindowFeatureComputer.h"


class ImageFeatureComputer {
public:
	ImageFeatureComputer(const Image& img, const Window& window);
	// Will be computed features in the directions specified
	// Default = 4 = all feautures ; oder 0->45->90->135°
    vector<WindowFeatures> computeAllFeatures(int numberOfDirections = 4);

	// This method will get all the feature names and all their values computed in the image
	vector<map<FeatureNames, vector<double>>> getAllDirectionsAllFeatureValues(const vector<WindowFeatures>& imageFeatures);
	// This method will print all the feature names and all their values computed in the image
	void printAllDirectionsAllFeatureValues(const vector<map<FeatureNames, vector<double>>>& featureList);

	// DEBUG, not really useful
	// Method will print, for each direction, for each window, all the features
	static void printImageAllDirectionsAllFeatures(const vector<WindowFeatures> &imageFeatureList);
	// Method will print, for each direction, for each window, all the features
	static void printImageAllDirectionsSingleFeature(const vector<WindowFeatures> &imageFeatureList, FeatureNames fname);

	vector<cv::Mat> generateFeatureImage(const vector<WindowFeatures> imageFeatures, FeatureNames fname);
	// TODO these methods might be unuseful; the user provides the number of directions it is interested to
	static void printImageSingleDirectionsAllFeatures(const vector<WindowFeatures> &imageFeatureList);
	static void printImageSingleDirectionsSingleFeature(const vector<WindowFeatures> &imageFeatureList, FeatureNames fname);
private:
	short int numberOfDirections;
	Image image;
	// Information to pass to WindowFeatureComputer equal to all generated windows
	// dimension, distance, symmetric while
	Window windowData;

};


#endif //FEATUREEXTRACTOR_IMAGEFEATURECOMPUTER_H
