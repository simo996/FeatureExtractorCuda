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
    FeatureComputer(const vector<int>& inputPixel, int maxGrayLevel, int shiftRows, int shiftColumns, Window windowData);
    map<FeatureNames, double> computeFeatures();
    // Support methods
    static void printFeatures(map<FeatureNames, double>& features);
    static void printGLCM(const GLCM& glcm); // print following information
    static void printGlcmData(const GLCM& glcm);
    static void printGlcmElements(const GLCM& glcm);
    static void printGlcmAggregated(const GLCM& glcm);
private:
    // given data to initialize GLCM
    vector<int> inputPixels;
    int maxGrayLevel;
    Window windowData;

    // Actual computation of all 18 features
    map<FeatureNames, double> computeBatchFeatures(const GLCM& metaGLCM);
    void extractAutonomousFeatures(const GLCM& metaGLCM, map<FeatureNames, double>& features);
    void extractSumAggregatedFeatures(const GLCM& metaGLCM, map<FeatureNames, double>& features);
    void extractDiffAggregatedFeatures(const GLCM& metaGLCM, map<FeatureNames, double>& features);
    void extractMarginalFeatures(const GLCM& metaGLCM, map<FeatureNames, double>& features);

};


#endif //FEATUREEXTRACTOR_FEATURECOMPUTER_H
