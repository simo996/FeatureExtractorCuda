//
// Created by simo on 11/07/18.
//

#ifndef FEATUREEXTRACTOR_FEATURECOMPUTER_H
#define FEATUREEXTRACTOR_FEATURECOMPUTER_H

#include <vector>
#include "GLCM.h"

enum FeatureNames { ASM, AUTOCORRELATION, ENTROPY, MAXPROB, HOMOGENEITY, CONTRAST,
    CORRELATION, CLUSTERPROMINENCE, CLUSTERSHADE, SUMOFSQUARES, DISSIMILARITY, IDM,
    SUMAVERAGE, SUMENTROPY, SUMVARIANCE, DIFFENTROPY, DIFFVARIANCE, IMOC
};

class FeatureComputer {
    /*
     * RESPONSABILITA CLASSE: Computare le 18 features per la singola direzione della finestra
     */

public:
    FeatureComputer(vector<int>& inputPixel, int distance, int shiftRows, int shiftColumns, int windowDimension, int maxGrayLevel, bool simmetric = false);
    map<FeatureNames, double> computeFeatures();
    // Support methods
    static void printFeatures(map<FeatureNames, double>& features);
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
    bool symmetric;
    // Actual computation of all 18 features
    map<FeatureNames, double> computeBatchFeatures(const GLCM& metaGLCM);
    void extractAutonomousFeatures(const GLCM& metaGLCM, map<FeatureNames, double>& features);
    void extractSumAggregatedFeatures(const GLCM& metaGLCM, map<FeatureNames, double>& features);
    void extractDiffAggregatedFeatures(const GLCM& metaGLCM, map<FeatureNames, double>& features);
    void extractMarginalFeatures(const GLCM& metaGLCM, map<FeatureNames, double>& features);

};


#endif //FEATUREEXTRACTOR_FEATURECOMPUTER_H
