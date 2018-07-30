/*
	Created by simo on 25/07/18.
    * RESPONSABILITA CLASSE: Scorrere l'immagine creando tutte le possibili finestre sovrapposte
*/

#ifndef FEATUREEXTRACTOR_IMAGEFEATURECOMPUTER_H
#define FEATUREEXTRACTOR_IMAGEFEATURECOMPUTER_H

#include "WindowFeatureComputer.h"


class ImageFeatureComputer {
public:
	ImageFeatureComputer(const Image& img, const Window& window);
	// TODO allow the user to choose in which directions the features will be computed and printed
	// Will be computed features in the directions specified
	// Default = 4 = all feautures ; oder 0->45->90->135°
    vector<WindowFeatures> computeAllFeatures(int numberOfDirections = 4);
	static void printImageAllDirectionsAllFeatures(const vector<WindowFeatures> &imageFeatureList);
	static void printImageAllDirectionsSingleFeature(const vector<WindowFeatures> &imageFeatureList, FeatureNames fname);
	static void printImageSingleDirectionsAllFeatures(const vector<WindowFeatures> &imageFeatureList);
	static void printImageSingleDirectionsSingleFeature(const vector<WindowFeatures> &imageFeatureList, FeatureNames fname);
private:
	Image image;
	// Information to pass to WindowFeatureComputer equal to all generated windows
	// dimension, distance, symmetric while
	Window windowData;

};


#endif //FEATUREEXTRACTOR_IMAGEFEATURECOMPUTER_H
