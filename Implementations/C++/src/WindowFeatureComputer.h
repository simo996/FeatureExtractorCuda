#ifndef FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H
#define FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H

#include "FeatureComputer.h"
#include "Direction.h"

typedef vector<double> WindowFeatures;
typedef vector<double> FeatureValues;

using namespace std;

/*
 * This class will compute the features for a direction of the window of interest
 */

class WindowFeatureComputer {

public:
    WindowFeatureComputer(unsigned int * pixels, const ImageData& img, const Window& wd, WorkArea& wa);
    // Will be computed features in the direction specified
    void computeWindowFeatures();
    /* Oss. No sense in computing a single feature, simply select the one
      needed from the complete list
     */
private:
    // Pixels of the image
    unsigned int * pixels;
    // Metadata about the image (dimensions, maxGrayLevel)
    ImageData image;
    // Metadata about the window
    Window windowData;
    // Memory location used for computation
    WorkArea& workArea;
};


#endif //FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H
