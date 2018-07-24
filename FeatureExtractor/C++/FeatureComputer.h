//
// Created by simo on 11/07/18.
//

#ifndef FEATUREEXTRACTOR_FEATURECOMPUTER_H
#define FEATUREEXTRACTOR_FEATURECOMPUTER_H

#include <vector>
#include "GLCM.h"


class FeatureComputer {
    /*
     * RESPONSABILITA CLASSE: Computare le 18 features per la singola direzione della finestra
     */

public:
    FeatureComputer(vector<int>& inputPixel, int distance, int shiftRows, int shiftColumns, int windowDimension, int maxGrayLevel, bool simmetric = false);
    map<string, double> computeFeatures();
    // Support methods
    static void printFeatures(map<std::string, double>& features);
    static void printGLCM(const GLCM& glcm); // print following information
    static void printGlcmData(const GLCM& glcm);
    static void printGlcmElements(const GLCM& glcm);
    static void printGlcmAggregated(const GLCM& glcm);
private:
    // Given Input
    vector<int> inputPixels;
    // Data to initialize given GLCM
    // TODO think about encapsulating in a struct
    int maxGrayLevel;
    int distance; // modulo tra reference e neighbor
    int windowDimension;
    int shiftRows;
    int shiftColumns;
    bool simmetric;
    // Actual computation of all 18 features
    map<string, double> computeBatchFeatures(const GLCM& metaGLCM);
    void extractAutonomousFeatures(const GLCM& metaGLCM, map<string, double>& features);
    void extractSumAggregatedFeatures(const GLCM& metaGLCM, map<string, double>& features);
    void extractDiffAggregatedFeatures(const GLCM& metaGLCM, map<string, double>& features);
    void extractMarginalFeatures(const GLCM& metaGLCM, map<string, double>& features);

};


#endif //FEATUREEXTRACTOR_FEATURECOMPUTER_H
