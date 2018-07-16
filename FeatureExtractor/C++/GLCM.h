//
// Created by simo on 11/07/18.
//

#ifndef FEATUREEXTRACTOR_GLCM_H
#define FEATUREEXTRACTOR_GLCM_H

#include <map>
#include "GrayPair.h"
#include "AggregatedGrayPair.h"

using namespace std;

class GLCM {
public:
    // Internal State
    map<GrayPair, int> grayPairsMap;
    // Getters method exposed for feature computer class
    int getNumberOfPairs() const;
    int getMaxGrayLevel() const;
    // Standard initializer constructor
    GLCM(int distance, int shiftRows, int shiftColumns, int windowDimension, int maxGrayLevel, bool simmetric = false);
    // Generate elements given the input
    void initializeElements(const vector<int>& inputPixels);
    // Utilities
    void printGLCMData() const;
    void printGLCMElements() const;

    // Representations useful for aggregated features
    // TODO think about this constness
    map<AggregatedGrayPair, int> codifySummedPairs() const;
    map<AggregatedGrayPair, int> codifySubtractedPairs() const;
    void printGLCMAggregatedElements(map<AggregatedGrayPair, int> input, bool areSummed);
    void printAggregated();
    // Representation useful for HXY
    map<int, int>codifyXMarginalProbabilities();
    map<int, int>codifyYMarginalProbabilities();
private:
    int maxGrayLevel;
    int numberOfPairs;
    int distance; // modulo tra reference e neighbor
    int windowDimension;
    int shiftRows;
    int shiftColumns;
    bool simmetric;

    // Addressing methods to get to neighbor pixel
    int computeColumnOffset();
    int computeRowOffset();
    // Geometric limits in the father windows
    int getBorderRows() const;
    int getBorderColumns() const;

};


#endif //FEATUREEXTRACTOR_GLCM_H
