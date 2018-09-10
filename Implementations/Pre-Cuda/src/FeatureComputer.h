#ifndef FEATUREEXTRACTOR_FEATURECOMPUTER_H
#define FEATUREEXTRACTOR_FEATURECOMPUTER_H

#include <vector>
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
    FeatureComputer(const unsigned int * pixels, const ImageData& img,
            int shiftRows, int shiftColumns, const Window& windowData,
            WorkArea& wa);
private:
    // given data to initialize related GLCM
    const unsigned int * pixels;
    ImageData image;
    // Window of interest
    Window windowData;
    // Memory location used for computing this window's feature
    WorkArea& workArea;
    // Where to put results
    double * featureOutput;
    // offset to indentify right index where to put results
    int outputWindowOffset;
    void computeOutputWindowFeaturesIndex();

    // Actual computation of all 18 features
    void computeDirectionalFeatures();
    void extractAutonomousFeatures(const GLCM& metaGLCM, double* features);
    void extractSumAggregatedFeatures(const GLCM& metaGLCM, double* features);
    void extractDiffAggregatedFeatures(const GLCM& metaGLCM, double* features);
    void extractMarginalFeatures(const GLCM& metaGLCM, double* features);

};

#endif //FEATUREEXTRACTOR_FEATURECOMPUTER_H
