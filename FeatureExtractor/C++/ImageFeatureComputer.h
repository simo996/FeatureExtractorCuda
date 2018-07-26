/*
	Created by simo on 25/07/18.
    * RESPONSABILITA CLASSE: Scorrere l'immagine creando tutte le possibili finestre sovrapposte
*/

#ifndef FEATUREEXTRACTOR_IMAGEFEATURECOMPUTER_H
#define FEATUREEXTRACTOR_IMAGEFEATURECOMPUTER_H

#include <vector>
#include <map>
#include "WindowFeatureComputer.h"
#include "Window.h"


struct ImageData{
	int rows;
	int columns;
	int maxGrayLevel;
};

class ImageFeatureComputer {
public:
	ImageFeatureComputer(const vector<int>& imagePixels, const ImageData, const Window);
	vector<WindowFeatures> computeAllFeatures();
private:
	vector<int> imagePixels;
	ImageData imgData;
	// Information to pass to WindowFeatureComputer = to all generated windows
	// dimension, distance, symmetric while
	Window windowData;

	vector<int>& locateWindowPixels(int i, int j, int windowDimension);

};


#endif //FEATUREEXTRACTOR_IMAGEFEATURECOMPUTER_H
