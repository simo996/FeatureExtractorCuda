/*
	Created by simo on 25/07/18.
    * RESPONSABILITA CLASSE: Scorrere l'immagine creando tutte le possibili finestre sovrapposte
*/

#ifndef FEATUREEXTRACTOR_IMAGEFEATURECOMPUTER_H
#define FEATUREEXTRACTOR_IMAGEFEATURECOMPUTER_H

#include "WindowFeatureComputer.h"
#include "Window.h"


class ImageFeatureComputer {
public:
	ImageFeatureComputer(const Image& img, const Window& window);
	vector<WindowFeatures> computeAllFeatures();
private:
	Image image;
	// Information to pass to WindowFeatureComputer = to all generated windows
	// dimension, distance, symmetric while
	Window windowData;

};


#endif //FEATUREEXTRACTOR_IMAGEFEATURECOMPUTER_H
