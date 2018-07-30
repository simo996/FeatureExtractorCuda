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
	vector<WindowFeatures> computeAllFeatures();
	static void printImageAllFeatures(const vector<WindowFeatures>& imageFeatureList);
	static void printImageSingleFeature(const vector<WindowFeatures>& imageFeatureList, FeatureNames fname);
private:
	Image image;
	// Information to pass to WindowFeatureComputer equal to all generated windows
	// dimension, distance, symmetric while
	Window windowData;

};


#endif //FEATUREEXTRACTOR_IMAGEFEATURECOMPUTER_H
