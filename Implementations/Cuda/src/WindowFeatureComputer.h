#ifndef WINDOWFEATURECOMPUTER_H_
#define WINDOWFEATURECOMPUTER_H_

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__ 
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif

#include "FeatureComputer.h"
#include "Direction.h"

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
    CUDA_DEV WindowFeatureComputer(unsigned int * pixels, const ImageData& img, const Window& wd, WorkArea& wa);
    /**
     * Computed features in the direction specified
     */
    CUDA_DEV void computeWindowFeatures();
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
#endif /* WINDOWFEATURECOMPUTER_H_ */
