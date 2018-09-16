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

/*
 * This class will compute the features for a direction of the window of interest
 */
class WindowFeatureComputer {

public:
    CUDA_DEV WindowFeatureComputer(unsigned int * pixels, const ImageData& img, const Window& wd, WorkArea& wa);
    // Will be computed features in the direction specified
    CUDA_DEV void computeWindowFeatures();
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
#endif /* WINDOWFEATURECOMPUTER_H_ */
