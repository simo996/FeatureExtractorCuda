//  Contiene le rappresentazione della GLCM utili per calcolare le features
// Created by simo on 11/07/18.
//

#ifndef FEATUREEXTRACTOR_GLCM_H
#define FEATUREEXTRACTOR_GLCM_H

#include <map>
#include "GrayPair.h"
#include "AggregatedGrayPair.h"
#include "Window.h"
#include "Image.h"


using namespace std;

class GLCM {
public:
    // Internal State
    map<GrayPair, int> grayPairsMap;
    // Standard initializer constructor
    GLCM(const Image& image, Window& windowData);
    // Utilities
    void printGLCMData() const;
    void printGLCMElements() const;
    // Getters method exposed for feature computer class
    int getNumberOfPairs() const;
    int getMaxGrayLevel() const;

    // Representations useful for aggregated features
    map<AggregatedGrayPair, int> codifySummedPairs() const;
    map<AggregatedGrayPair, int> codifySubtractedPairs() const;
    void printGLCMAggregatedElements(map<AggregatedGrayPair, int> input, bool areSummed) const;
    void printAggregated() const;
    // Representation useful for HXY
    map<int, int> codifyXMarginalProbabilities() const;
    map<int, int> codifyYMarginalProbabilities() const;
    // Rendere statico
    static void printMarginalProbability(map<int, int> marginalProb, char symbol);

    private:
    // TODO think about encapsulating in a struct initialization data
    Image img;
    int numberOfPairs;
    Window windowData;

    // Addressing methods to get to neighbor pixel
    int computeWindowColumnOffset();
    int computeWindowRowOffset();
    // Geometric limits in the father windows
    int getBorderRows() const;
    int getBorderColumns() const;
    // Methods to build the glcm from input pixel and directional data
    void initializeGlcmElements();
    int getReferenceIndex(int i, int j, int initialRowOffset, int initialColumnOffset);
    int getNeighborIndex(int i, int j, int initialColumnOffset);

};


#endif //FEATUREEXTRACTOR_GLCM_H
