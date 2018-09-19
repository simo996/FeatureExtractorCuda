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

/**
 * This class will compute 18 features for a single window, for a
 * particular direction
 */
class FeatureComputer {
public:
   /**
     * Initialize the object and computate the features of interest to this
     * object
     * @param pixels: pixels of the entire image
     * @param img: metadata about the image (physical dimensions,
     * maxGrayLevel, borders)
     * @param shiftRows: shift on the y-axis to apply to locate the neighbor
     * pixel
     * @param shiftColumns: shift on the x-axis to apply to locate the neighbor
     * pixel
     * @param windowData: metadata about the window of interest (size, starting
     * point in the image, etc.)
     * @param wa: memory location where this object will create the arrays of
     * representation needed for computing its features
     */
    CUDA_DEV FeatureComputer(const unsigned int * pixels, const ImageData& img,
            int shiftRows, int shiftColumns, const Window& windowData,
            WorkArea& wa);
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
     * Window of interest where the features are computed
     */
    Window windowData;
    /**
     * Memory location used for computing this window's feature
     */
    WorkArea& workArea;
    /**
     * Where to put results
     */
    double * featureOutput;
    /**
     * offset to identify the window that is being computed by yhe
     * object; this information will be used for storing the results in the
     * correct memory location
     */
    int outputWindowOffset;
    /**
     * Compute the offset to identify the window that is being computed by yhe
     * object; this information will be used for storing the results in the
     * correct memory location
     */
    CUDA_DEV void computeOutputWindowFeaturesIndex();

    /**
     * Launch computation of all features supported
     */
    CUDA_DEV void computeDirectionalFeatures();
    /**
     * Compute the features that can be extracted from the GLCM of the image;
     * this method will store the results automatically
     * @param metaGLCM: object of class GLCM that will provide gray pairs
     * @param features: where to store the results; this pointer is obtained
     * from the work area
     */
    CUDA_DEV void extractAutonomousFeatures(const GLCM& metaGLCM, double* features);
    /**
     * Compute the features that can be extracted from the AggregatedPairs
     * obtained by adding gray levels of the pixel pairs.
     * this method will store the results automatically
     * @param metaGLCM: object of class GLCM that will provide gray pairs
     * @param features: where to store the results; this pointer is obtained
     * from the work area
     */
    CUDA_DEV void extractSumAggregatedFeatures(const GLCM& metaGLCM, double* features);
    /**
     * Compute the features that can be extracted from the AggregatedPairs
     * obtained by subtracting gray levels of the pixel pairs.
     * this method will store the results automatically
     * @param metaGLCM: object of class GLCM that will provide gray pairs
     * @param features: where to store the results; this pointer is obtained
     * from the work area
     */
    CUDA_DEV void extractDiffAggregatedFeatures(const GLCM& metaGLCM, double* features);
    /**
     * Compute the features that can be extracted from the AggregatedPairs
     * obtained by computing the marginal frequency of the gray levels of the
     * reference/neighbor pixels.
     * this method will store the results automatically
     * @param metaGLCM: object of class GLCM that will provide gray pairs
     * @param features: where to store the results; this pointer is obtained
     * from the work area
     */
    CUDA_DEV void extractMarginalFeatures(const GLCM& metaGLCM, double* features);
};


#endif /* FEATURECOMPUTER_H_ */
