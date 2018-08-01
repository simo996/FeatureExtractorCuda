//
// Created by simo on 11/07/18.
//

#ifndef FEATUREEXTRACTOR_FEATURECOMPUTER_H
#define FEATUREEXTRACTOR_FEATURECOMPUTER_H

#include <vector>
#include "GLCM.h"
#include "Features.h"

class FeatureComputer {
    /*
     * RESPONSABILITA CLASSE: Computare le 18 features per la singola direzione della finestra
     * Espone metodi per stampare i risultati
     */

public:
    FeatureComputer(const Image& img, int shiftRows, int shiftColumns, const Window& windowData);
    map<FeatureNames, double> computeDirectionalFeatures();
    // Methods for printing features and their label

private:
    // given data to initialize GLCM
    Image image;
    Window windowData;

    // Actual computation of all 18 features
    map<FeatureNames, double> computeBatchFeatures(const GLCM& metaGLCM);
    void extractAutonomousFeatures(const GLCM& metaGLCM, map<FeatureNames, double>& features);
    void extractSumAggregatedFeatures(const GLCM& metaGLCM, map<FeatureNames, double>& features);
    void extractDiffAggregatedFeatures(const GLCM& metaGLCM, map<FeatureNames, double>& features);
    void extractMarginalFeatures(const GLCM& metaGLCM, map<FeatureNames, double>& features);

    // Support methods useful for debug purpose
    static void printGLCM(const GLCM& glcm); // prints following information
    static void printGlcmData(const GLCM& glcm);
    static void printGlcmElements(const GLCM& glcm);
    static void printGlcmAggregated(const GLCM& glcm);
};


#endif //FEATUREEXTRACTOR_FEATURECOMPUTER_H
