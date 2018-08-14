//  Contiene le rappresentazione della GLCM utili per calcolare le features
// Created by simo on 11/07/18.
//

#ifndef FEATUREEXTRACTOR_GLCM_H
#define FEATUREEXTRACTOR_GLCM_H

#include <map>
#include "GrayPair.h"
#include "AggregatedGrayPair.h"
#include "Window.h"
#include "ImageData.h"


using namespace std;

class GLCM {
public:
    vector<GrayPair> elements;
    vector<AggregatedGrayPair> summedPairs;
    vector<AggregatedGrayPair> subtractedPairs;
    vector<AggregatedGrayPair> xMarginalPairs;
    vector<AggregatedGrayPair> yMarginalPairs;

    unsigned int getNumberOfUniquePairs() const;
    unsigned int getNumberOfUniqueAggregatedElements(const vector<AggregatedGrayPair>& src) const;
    unsigned int getNumberOfUniqueMarginalElements(const vector<unsigned int>& src);

    // Standard initializer constructor
    GLCM(const unsigned int * pixels, const ImageData& image, Window& windowData);

    // Utilities
    void printGLCMData() const;
    void printGLCMElements() const;
    void printAggregated() const;
    void printMarginalProbabilityElements() const;

    // Getters method exposed for feature computer class
    int getNumberOfPairs() const;
    int getMaxGrayLevel() const;

private:
    const unsigned int * pixels;
    ImageData img;

    // WARNING, SE SIMMETRICA LO SPAZIO DEVE RADDOPPIARE
    int numberOfPairs;
    Window windowData;

    // Addressing methods to get to neighbor pixel
    int computeWindowColumnOffset();
    int computeWindowRowOffset();
    // Geometric limits in the father windows
    int getBorderRows() const;
    int getBorderColumns() const;
    int getReferenceIndex(int i, int j, int initialRowOffset, int initialColumnOffset);
    int getNeighborIndex(int i, int j, int initialColumnOffset);
    // Methods to build the glcm from input pixel and directional data
    void initializeGlcmElements();
    // Representations useful for aggregated features
    void codifySummedPairs();
    void codifySubtractedPairs();
    // Representation useful for HXY
    void codifyXMarginalProbabilities() ;
    void codifyYMarginalProbabilities() ;
    // debug printing methods
    void printGLCMAggregatedElements(bool areSummed) const;

};


#endif //FEATUREEXTRACTOR_GLCM_H
