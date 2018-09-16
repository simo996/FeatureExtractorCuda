#ifndef FEATURECOMPUTER_H_
#define FEATURECOMPUTER_H_

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__ 
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif

#include "GLCM.h"
#include "Features.h"

/*
 * This class will compute 18 features for a single window, for a
 * particular direction
 */

class FeatureComputer {
public:
    /* Initialize the data structures needed; computes the features
     * saving the results in the right spot of the given output vector
     */
    CUDA_DEV FeatureComputer(const unsigned int * pixels, const ImageData& img,
            int shiftRows, int shiftColumns, const Window& windowData,
            WorkArea& wa);
private:
    // given data to initialize related GLCM
    // Pixels of the image
    const unsigned int * pixels;
    // Metadata about the image (dimensions, maxGrayLevel)
    ImageData image;
    // Window of interest
    Window windowData;
    // Memory location used for computing this window's feature
    WorkArea& workArea;
    // Where to put results
    double * featureOutput;
    // offset to indentify right index where to put results
    int outputWindowOffset;
    CUDA_DEV void computeOutputWindowFeaturesIndex();

    // Actual computation of all 18 features
    CUDA_DEV void computeDirectionalFeatures();
    CUDA_DEV void extractAutonomousFeatures(const GLCM& metaGLCM, double* features);
    CUDA_DEV void extractSumAggregatedFeatures(const GLCM& metaGLCM, double* features);
    CUDA_DEV void extractDiffAggregatedFeatures(const GLCM& metaGLCM, double* features);
    CUDA_DEV void extractMarginalFeatures(const GLCM& metaGLCM, double* features);
};


#endif /* FEATURECOMPUTER_H_ */
