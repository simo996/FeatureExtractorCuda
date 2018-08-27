/*
 * GLCM.h
 *
 *  Created on: 26/ago/2018
 *      Author: simone
 */

#ifndef GLCM_H_
#define GLCM_H_

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__ 
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif

#include "GrayPair.h"
#include "AggregatedGrayPair.h"
#include "Window.h"
#include "ImageData.h"
#include "WorkArea.h"


using namespace std;

class GLCM {
public:
    GrayPair* elements;
    int effectiveNumberOfGrayPairs;
    AggregatedGrayPair* summedPairs;
    int numberOfSummedPairs;
    AggregatedGrayPair* subtractedPairs;
    int numberOfSubtractedPairs;
    AggregatedGrayPair* xMarginalPairs;
    int numberOfxMarginalPairs;
    AggregatedGrayPair* yMarginalPairs;
    int numberOfyMarginalPairs;

    // Standard initializer constructor
    CUDA_DEV GLCM(const unsigned int * pixels, const ImageData& image, Window& windowData, WorkArea& wa);
    CUDA_DEV ~GLCM();

    // Getters method exposed for feature computer class
    CUDA_DEV int getNumberOfPairs() const;
    CUDA_DEV int getMaxGrayLevel() const;

    // Utilities
    CUDA_DEV void printGLCM() const;

private:
    const unsigned int * pixels;
    ImageData img;
    int numberOfPairs;
    Window windowData;
    WorkArea& workArea;

    // Addressing methods to get to neighbor pixel
    CUDA_DEV int computeWindowColumnOffset();
    CUDA_DEV int computeWindowRowOffset();
    // Geometric limits in the father windows
    CUDA_DEV int getWindowRowsBorder() const;
    CUDA_DEV int getWindowColsBorder() const;
    CUDA_DEV int getReferenceIndex(int i, int j, int initialRowOffset, int initialColumnOffset);
    CUDA_DEV int getNeighborIndex(int i, int j, int initialColumnOffset);
    // Methods to build the glcm from input pixel and directional data
    CUDA_DEV void insertElement(GrayPair* elements, GrayPair actualPair,
            uint& lastInsertionPosition);
    CUDA_DEV void insertElement(AggregatedGrayPair* elements,
            AggregatedGrayPair actualPair, uint& lastInsertionPosition);
    CUDA_DEV void initializeGlcmElements();
    // Representations useful for aggregated features
    CUDA_DEV void codifyAggregatedPairs();
    // Representation useful for HXY
    CUDA_DEV void codifyMarginalPairs() ;

    // debug printing methods
    CUDA_DEV void printGLCMData() const;
    CUDA_DEV void printGLCMElements() const;
    CUDA_DEV void printAggregated() const;
    CUDA_DEV void printMarginalProbabilityElements() const;
    CUDA_DEV void printGLCMAggregatedElements(bool areSummed) const;
};
#endif /* GLCM_H_ */
