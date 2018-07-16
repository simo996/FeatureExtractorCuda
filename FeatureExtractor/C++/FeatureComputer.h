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
    map<std::string, double> computeFeatures();
private:

    // Given Input
    vector<int> inputPixels;
    // Data to initialize given GLCM
    int maxGrayLevel;
    int distance; // modulo tra reference e neighbor
    int windowDimension;
    int shiftRows;
    int shiftColumns;
    bool simmetric;
    // Actual computation of all 18 features
    map<string, double> extractFeatures(const GLCM& glcm);

    // Single feature extraction
    double computeASM(const GLCM& glcm);
    double computeAutocorrelation(const GLCM& glcm);
    double computeEntropy(const GLCM& glcm);
    double computeMaximumProbability(const GLCM& glcm);
    double computeHomogeneity(const GLCM& glcm);
    double computeContrast(const GLCM& glcm);
    double computeDissimilarity(const  GLCM& metaGLCM);
    double computeInverceDifferentMoment(const GLCM& glcm);
    double computeCorrelation(const GLCM& glcm, double muX, double muY, double sigmaX, double sigmaY);

    double computeClusterProminence(const GLCM& glcm, double muX, double muY);
    double computeClusterShade(const GLCM& glcm, double muX, double muY);
    double computeSumOfSquares(const GLCM& glcm, double mu);
    double computeSumAverage(const map<AggregatedGrayPair, int>& summedMetaGLCM, int numberOfPairs);

    double computeSumEntropy(const map<AggregatedGrayPair, int>& summedMetaGLCM, int numberOfPairs);
    double computeSumVariance(const map<AggregatedGrayPair, int>& summedMetaGLCM, double sumEntropy, int numberOfPairs);
    double computeDifferenceEntropy(const map<AggregatedGrayPair, int>& aggregatedMetaGLCM, int numberOfPairs);
    double computeDifferenceVariance(const map<AggregatedGrayPair, int>& aggregatedMetaGLCM, int numberOfPairs);

    double computeMean(const GLCM& glcm);
    double computeMuX(const GLCM& glcm);
    double computeMuY(const GLCM& glcm);
    double computeSigmaX(const GLCM& glcm, double muX);
    double computeSigmaY(const GLCM& glcm, double muY);

    // Support methods
    void printFeatures(map<std::string, double>& features);
    void printGLCM(const GLCM& glcm); // print following information
    void printGlcmData(const GLCM& glcm);
    void printGlcmElements(const GLCM& glcm);
    void printGlcmAggregated(const GLCM& glcm);
};


#endif //FEATUREEXTRACTOR_FEATURECOMPUTER_H
