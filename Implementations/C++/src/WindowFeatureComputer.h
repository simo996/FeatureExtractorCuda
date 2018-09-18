#ifndef FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H
#define FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H

#include "FeatureComputer.h"
#include "Direction.h"

/**
 * Array of all the features that can be extracted simultaneously from a
 * window
 */
typedef vector<double> WindowFeatures;
/**
 * Array of all the features that can be extracted simultaneously from a
 * direction in a window
 */
typedef vector<double> FeatureValues;

using namespace std;

/**
 * This class will compute the features for a direction of the window of interest
 */
class WindowFeatureComputer {

public:
    /**
     * Construct the class that will compute the features for a window
     * @param pixels: of the entire image
     * @param img: metadata about the image (physical dimensions,
     * maxGrayLevel, borders)
     * @param wd: metadata about the window of interest (size, starting
     * point in the image)
     * @param wa: memory location where this object will create the arrays of
     * representation needed for computing its features
     */
    WindowFeatureComputer(unsigned int * pixels, const ImageData& img, const Window& wd, WorkArea& wa);
    /**
     * Computed features in the direction specified
     */
    void computeWindowFeatures();
    /* Oss. No sense in computing a single feature, simply select the one
      needed from the complete list
     */
private:
    /**
     * Pixels of the image
     */
    const unsigned int * pixels;
    /**
     * Metadata about the image (dimensions, maxGrayLevel)
     */
    ImageData image;
    /**
     * Window of interest where the glcm is computed
     */
    Window windowData;
    /**
     * Memory location used for computing this window's feature
     */
    WorkArea& workArea;
};


#endif //FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H
