/*
 * FeatureComputer.h
 *
 *  Created on: 26/ago/2018
 *      Author: simone
 */

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

class FeatureComputer {
    /*
     * RESPONSABILITA CLASSE: Computare le 18 features per la singola direzione della finestra
     * Espone metodi per stampare i risultati
     */

public:
    CUDA_DEV FeatureComputer(const unsigned int * pixels, const ImageData& img,
            int shiftRows, int shiftColumns, const Window& windowData,
            WorkArea& wa, short int directionNumber);
    CUDA_DEV void computeDirectionalFeatures();
private:
    // given data to initialize related GLCM
    const unsigned int * pixels;
    ImageData image;
    Window windowData;
    WorkArea& workArea;
    // offset to indentify where to put results
    short int directionOffset;
    int outputWindowOffset;
    double * featureOutput;

    // Actual computation of all 18 features
    CUDA_DEV void extractAutonomousFeatures(const GLCM& metaGLCM, double* features);
    CUDA_DEV void extractSumAggregatedFeatures(const GLCM& metaGLCM, double* features);
    CUDA_DEV void extractDiffAggregatedFeatures(const GLCM& metaGLCM, double* features);
    CUDA_DEV void extractMarginalFeatures(const GLCM& metaGLCM, double* features);

    CUDA_DEV void computeOutputWindowFeaturesIndex();
    // Support method useful for debugging this class
    CUDA_DEV static void printGLCM(const GLCM& glcm); // prints glcms various information
};


#endif /* FEATURECOMPUTER_H_ */
