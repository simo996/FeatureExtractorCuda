//  Contiene le rappresentazione della GLCM utili per calcolare le features
// Created by simo on 11/07/18.
//

#ifndef FEATUREEXTRACTOR_GLCM_H
#define FEATUREEXTRACTOR_GLCM_H

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
    GLCM(const unsigned int * pixels, const ImageData& image, Window& windowData, WorkArea& wa);
    ~GLCM();

    // Getters method exposed for feature computer class
    int getNumberOfPairs() const;
    int getMaxGrayLevel() const;

    // Utilities
    void printGLCM() const;

private:
    const unsigned int * pixels;
    ImageData img;
    int numberOfPairs;
    Window windowData;
    WorkArea& workArea;

    // Addressing methods to get to neighbor pixel
    int computeWindowColumnOffset();
    int computeWindowRowOffset();
    // Geometric limits in the father windows
    int getWindowRowsBorder() const;
    int getWindowColsBorder() const;
    int getReferenceIndex(int i, int j, int initialRowOffset, int initialColumnOffset);
    int getNeighborIndex(int i, int j, int initialColumnOffset);
    // Methods to build the glcm from input pixel and directional data
    void insertElement(GrayPair* elements, GrayPair actualPair,
            uint& lastInsertionPosition);
    void insertElement(AggregatedGrayPair* elements,
            AggregatedGrayPair actualPair, uint& lastInsertionPosition);
    void initializeGlcmElements();
    // Representations useful for aggregated features
    void codifyAggregatedPairs();
    // Representation useful for HXY
    void codifyMarginalPairs() ;

    // debug printing methods
    void printGLCMData() const;
    void printGLCMElements() const;
    void printAggregated() const;
    void printMarginalProbabilityElements() const;
    void printGLCMAggregatedElements(bool areSummed) const;

};


#endif //FEATUREEXTRACTOR_GLCM_H
