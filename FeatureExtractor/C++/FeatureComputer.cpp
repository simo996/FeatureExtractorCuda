//
// Created by simo on 11/07/18.
//

#include <iostream>
#include <string>
#include <cmath>
#include "FeatureComputer.h"

using namespace std;

FeatureComputer::FeatureComputer(vector<int>& inputPixel, int distance, int shiftRows, int shiftColumns, int windowDimension, int maxGrayLevel,
                                 bool simmetric) {
    this->distance = distance;
    this->shiftRows = shiftRows;
    this->shiftColumns = shiftColumns;
    this->windowDimension = windowDimension;
    this->maxGrayLevel = maxGrayLevel;
    this->simmetric = simmetric;
    this->inputPixels = inputPixel;
}

map<string, double> FeatureComputer::computeFeatures() {
    GLCM glcm(distance, shiftRows, shiftColumns, windowDimension, maxGrayLevel, simmetric);
    glcm.initializeElements(inputPixels);
    printGLCM(glcm); // Print data and elements for debugging
    map<string, double> features = computeBatchFeatures(glcm);
    //printFeatures(features);
    return features;
}

void FeatureComputer::printFeatures(map<string, double>& features){
    cout << endl;
    // TODO think about moving from string to enum for accessing features
    cout << "ASM: \t" << features["ASM"] << endl;
    cout << "AUTOCORRELATION: \t" << features["AUTOCORRELATION"] << endl;
    cout << "ENTROPY: \t" << features["ENTROPY"] << endl;
    cout << "MAXIMUM PROBABILITY: \t" << features["MAXPROB"] << endl;
    cout << "HOMOGENEITY: \t" << features["HOMOGENEITY"] << endl;
    cout << "CONTRAST: \t" << features["CONTRAST"] << endl;
    cout << "DISSIMILARITY: \t" << features["DISSIMILARITY"] << endl;

    cout << "CORRELATION: \t" << features["CORRELATION"] << endl;
    cout << "CLUSTER Prominence: \t" << features["CLUSTER PROMINENCE"] << endl;
    cout << "CLUSTER SHADE: \t" << features["CLUSTER SHADE"] << endl;
    cout << "SUM OF SQUARES: \t" << features["SUM OF SQUARES"] << endl;
    cout << "IDM normalized: \t" << features["IDM"] << endl;

    cout << "SUM AVERAGE: \t" << features["SUM AVERAGE"] << endl;
    cout << "SUM ENTROPY: \t" << features["SUM ENTROPY"] << endl;
    cout << "SUM VARIANCE: \t" << features["SUM VARIANCE"] << endl;

    cout << "DIFF ENTROPY: \t" << features["DIFF ENTROPY"] << endl;
    cout << "DIFF VARIANCE: \t" << features["DIFF VARIANCE"] << endl;

    cout << "INFORMATION MEASURE OF CORRELATION: \t" << features["IMOC"] << endl;

    cout << endl;

}

/* TODO remove METHODS FOR DEBUG */
void FeatureComputer::printGLCM(const GLCM& glcm){
    printGlcmData(glcm);
    printGlcmElements(glcm);
    //printGlcmAggregated(glcm);
}

void FeatureComputer::printGlcmData(const GLCM& glcm){
    glcm.printGLCMData();
}
void FeatureComputer::printGlcmElements(const GLCM& glcm){
    glcm.printGLCMElements();
}
void FeatureComputer::printGlcmAggregated(const GLCM& glcm){
    glcm.printAggregated();
}

// ASM
inline double computeAsmStep(const double actualPairProbability){
    return pow((actualPairProbability),2);
}


// AUTOCORRELATION
inline double computeAutocorrelationStep(const int i, const int j, const double actualPairProbability){
    return (i * j * actualPairProbability);
}


// ENTROPY
inline double computeEntropyStep(const double actualPairProbability){
    return (actualPairProbability * log(actualPairProbability));
}


// HOMOGENEITY
inline double computeHomogeneityStep(const int i, const int j, const double actualPairProbability){
    return (actualPairProbability / (1 + fabs(i - j)));
}

// CONTRAST
inline double computeContrastStep(const int i, const int j, const double actualPairProbability){
    return (actualPairProbability * (pow(fabs(i - j), 2)));
}

// DISSIMILARITY
inline double computeDissimilarityStep(const int i, const int j, const double pairProbability){
    return (pairProbability * (fabs(i - j)));
}

// IDM
inline double computeInverceDifferenceMomentStep(const int i, const int j, 
    const double pairProbability, const int maxGrayLevel)
{
    return (pairProbability / (1 + fabs(i - j) / maxGrayLevel));
}

/* FEATURES WITH MEANS */
// CORRELATION
inline double computeCorrelationStep(const int i, const int j, 
    const double pairProbability, const double muX, const double muY, 
    const double sigmaX, const double sigmaY)
{
    //cout << "\ni: " << i << "\tj: " << j << "\t pariprob: " << pairProbability << "\t(i-muX): " << (i -muX) << "\t(j-muY): " << (j - muY) << "\tcontributo: " << (((i - muX) * (j - muY) * pairProbability ) / (sigmaX * sigmaY));
    return (((i - muX) * (j - muY) * pairProbability ) / (sigmaX * sigmaY));
}

// CLUSTER PROMINENCE
inline double computeClusterProminenceStep(const int i, const int j, 
    const double pairProbability, const double muX, const double muY){
    return (pow((i + j - muX - muY), 4) * pairProbability);
}

// CLUSTER SHADE
inline double computeClusterShadeStep(const int i, const int j,
    const double pairProbability, const double muX, const double muY){
    return (pow((i + j - muX - muY), 3) * pairProbability);
}

// SUM OF SQUARES
inline double computeSumOfSquaresStep(const int i,
                                      const double pairProbability, const double mean){
    //cout << (pow((i - mean), 2) * pairProbability);
    return (pow((i - mean), 2) * pairProbability);
}

// SUM Aggregated features
// SUM AVERAGE
inline double computeSumAverageStep(const double aggregatedGrayLevel, const double pairProbability){
    return (aggregatedGrayLevel * pairProbability);
}

// SUM ENTROPY
inline double computeSumEntropyStep(const double pairProbability){
    return (log(pairProbability) * pairProbability);
}

// SUM VARIANCE
inline double computeSumVarianceStep(const int aggregatedGrayLevel, 
    const double pairProbability, const double sumEntropy){
    return (pow((aggregatedGrayLevel - sumEntropy),2) * pairProbability);
}

// DIFF Aggregated features
// DIFF ENTROPY
inline double computeDiffEntropyStep(const double pairProbability){
    return (log(pairProbability) * pairProbability);
}

// DIFF
inline double computeDiffVarianceStep(const int aggregatedGrayLevel, const double pairProbability){
    return (pow(aggregatedGrayLevel, 2) * pairProbability);
}


// Marginal Features
inline double computeHxStep(const double grayLevelProbability){
    return (grayLevelProbability * log(grayLevelProbability));
}

inline double computeHyStep(const double grayLevelProbability){
    return (grayLevelProbability * log(grayLevelProbability));
}



map<string, double> FeatureComputer::computeBatchFeatures(const GLCM& glcm) {
    map<string, double> features;

    // Features computable from glcm Elements
    extractAutonomousFeatures(glcm, features);

    // Feature computable from aggregated glcm pairs
    extractSumAggregatedFeatures(glcm, features);
    extractDiffAggregatedFeatures(glcm, features);

    // Imoc
    extractMarginalFeatures(glcm, features);

    return features;
}

void FeatureComputer::extractAutonomousFeatures(const GLCM& glcm, map<string, double>& features){
    // Intermediate values
    double mean = 0;
    double muX = 0;
    double muY = 0;
    double sigmaX = 0;
    double sigmaY = 0;

    // Actual features
    double ASM = 0;
    double AUTOCORRELATION = 0;
    double ENTROPY = 0;
    double MAXPROB = 0;
    double HOMOGENEITY = 0;
    double CONTRAST = 0;
    double DISSIMILARITY = 0;
    double IDM = 0;

    // First batch of computable features
    typedef map<GrayPair, int>::const_iterator MI;

    for(MI actual=glcm.grayPairsMap.begin() ; actual != glcm.grayPairsMap.end(); actual++)
    {
        GrayPair actualPair = actual->first;
        int i = actualPair.getGrayLevelI();
        int j = actualPair.getGrayLevelJ();
        double actualPairProbability = ((double) actual->second)/glcm.getNumberOfPairs();

        ASM += computeAsmStep(actualPairProbability);
        AUTOCORRELATION += computeAutocorrelationStep(i, j, actualPairProbability);
        ENTROPY += computeEntropyStep(actualPairProbability);
        if(MAXPROB < actualPairProbability)
            MAXPROB = actualPairProbability;
        HOMOGENEITY += computeHomogeneityStep(i, j, actualPairProbability);
        CONTRAST += computeContrastStep(i, j, actualPairProbability);
        DISSIMILARITY += computeDissimilarityStep(i, j, actualPairProbability);
        IDM += computeInverceDifferenceMomentStep(i, j, actualPairProbability, glcm.getMaxGrayLevel());

        // intemediate values
        mean += (i * j * actualPairProbability);
        muX += (i * actualPairProbability);
        muY += (j * actualPairProbability);
    }

    features["ASM"]= ASM;
    features["AUTOCORRELATION"]= AUTOCORRELATION;
    features["ENTROPY"]= (-1 * ENTROPY);
    features["MAXPROB"]= MAXPROB;
    features["HOMOGENEITY"]= HOMOGENEITY;
    features["CONTRAST"]= CONTRAST;
    features["DISSIMILARITY"]= DISSIMILARITY;
    features["IDM"]= IDM;

    // Second batch of computable features
    double CLUSTERPROM = 0;
    double CLUSTERSHADE = 0;
    double SUMOFSQUARES = 0;

    for(MI actual=glcm.grayPairsMap.begin() ; actual != glcm.grayPairsMap.end(); actual++)
    {
        GrayPair actualPair = actual->first;
        int i = actualPair.getGrayLevelI();
        int j = actualPair.getGrayLevelJ();
        double actualPairProbability = ((double) actual->second)/glcm.getNumberOfPairs();

        CLUSTERPROM += computeClusterProminenceStep(i, j, actualPairProbability, muX, muY);
        CLUSTERSHADE += computeClusterShadeStep(i, j, actualPairProbability, muX, muY);
        SUMOFSQUARES += computeSumOfSquaresStep(i, actualPairProbability, mean);
        sigmaX += pow((i - muX), 2) * actualPairProbability;
        sigmaY+= pow((j - muY), 2) * actualPairProbability;
    }

    features["CLUSTER PROMINENCE"]= CLUSTERPROM;
    features["CLUSTER SHADE"]= CLUSTERSHADE;
    features["SUM OF SQUARES"]= SUMOFSQUARES;

    // Only feature that needs the third scan of the glcm
    double CORRELATION = 0;

    for(MI actual=glcm.grayPairsMap.begin() ; actual != glcm.grayPairsMap.end(); actual++)
    {
        GrayPair actualPair = actual->first;
        int i = actualPair.getGrayLevelI();
        int j = actualPair.getGrayLevelJ();
        double actualPairProbability = ((double) actual->second)/glcm.getNumberOfPairs();

        CORRELATION += computeCorrelationStep(i, j, actualPairProbability, 
            muX, muY, sigmaX, sigmaY);
    }
    features["CORRELATION"]= CORRELATION;
   


}

void FeatureComputer::extractSumAggregatedFeatures(const GLCM& glcm, map<string, double>& features) {
    map<AggregatedGrayPair, int> summedPairs = glcm.codifySummedPairs();
    int numberOfPairs = glcm.getNumberOfPairs();

    double SUMAVG = 0;
    double SUMENTROPY = 0;

    typedef map<AggregatedGrayPair, int>::const_iterator MI;
    for (MI actual = summedPairs.begin(); actual != summedPairs.end(); actual++) 
    {
        AggregatedGrayPair actualPair = actual->first;
        int k = actualPair.getAggregatedGrayLevel();
        double actualPairProbability = ((double) actual->second) / numberOfPairs;

        SUMAVG += computeSumAverageStep(k, actualPairProbability);
        SUMENTROPY += computeSumEntropyStep(actualPairProbability);
    }
    SUMENTROPY *= -1;
    features["SUM AVERAGE"] = SUMAVG;
    features["SUM ENTROPY"] = SUMENTROPY;

    double SUMVARIANCE = 0;
    for (MI actual = summedPairs.begin(); actual != summedPairs.end(); actual++)
    {
        AggregatedGrayPair actualPair = actual->first;
        int k = actualPair.getAggregatedGrayLevel();
        double actualPairProbability = ((double) actual->second) / numberOfPairs;

        SUMVARIANCE += computeSumVarianceStep(k, actualPairProbability, SUMENTROPY);
    }

    features["SUM VARIANCE"] = SUMVARIANCE;
}


void FeatureComputer::extractDiffAggregatedFeatures(const GLCM& glcm, map<string, double>& features) {
    map<AggregatedGrayPair, int> subtractedPairs = glcm.codifySubtractedPairs();
    int numberOfPairs= glcm.getNumberOfPairs();

    double DIFFENTROPY = 0;
    double DIFFVARIANCE = 0;

    typedef map<AggregatedGrayPair, int>::const_iterator MI;
    for (MI actual = subtractedPairs.begin(); actual != subtractedPairs.end(); actual++) 
    {
        AggregatedGrayPair actualPair = actual->first;
        int k = actualPair.getAggregatedGrayLevel();
        double actualPairProbability = ((double) actual->second) / numberOfPairs;

        DIFFENTROPY += computeDiffEntropyStep(actualPairProbability);
        DIFFVARIANCE += computeDiffVarianceStep(k, actualPairProbability);
    }
    DIFFENTROPY *= -1;
    features["DIFF ENTROPY"] = DIFFENTROPY;
    features["DIFF VARIANCE"] = DIFFVARIANCE;

}

void FeatureComputer::extractMarginalFeatures(const GLCM& glcm, map<string, double>& features){
    map<int, int> marginalPairsX = glcm.codifyXMarginalProbabilities();
    map<int, int> marginalPairsY = glcm.codifyYMarginalProbabilities();
    int numberOfPairs = glcm.getNumberOfPairs();
    double HX = 0;

    // Compute first intermediate value
    typedef map<int, int>::const_iterator MI;
    for (MI actual = marginalPairsX.begin(); actual != marginalPairsX.end(); actual++)
    {

        double probability = ((double) (actual->second)/numberOfPairs);
       
        HX += computeHxStep(probability);
    }
    HX *= -1;

    // Compute second intermediate value
    double HY = 0;
    for (MI actual = marginalPairsY.begin(); actual != marginalPairsY.end(); actual++)
    {

        double probability = ((double) (actual->second)/numberOfPairs);
       
        HY += computeHyStep(probability);
    }
    HY *= -1;

    // Extract third intermediate value
    double HXY = features["ENTROPY"];

    // Compute last intermediate value
    double HXY1 = 0;

    typedef map<GrayPair, int>::const_iterator MGPI;
    for (MGPI actual = glcm.grayPairsMap.begin(); actual != glcm.grayPairsMap.end(); actual++)
    {
        GrayPair actualPair = actual->first;
        int i = actualPair.getGrayLevelI();
        int j = actualPair.getGrayLevelJ();
        double actualPairProbability = ((double) actual->second) / numberOfPairs;
        double xMarginalProbability = (double) marginalPairsX.find(i)->second / numberOfPairs;
        double yMarginalProbability = (double) marginalPairsY.find(j)->second / numberOfPairs;

        HXY1 += actualPairProbability * log(xMarginalProbability * yMarginalProbability);
    }
    HXY1 *= -1;

    double IMOC = (HXY - HXY1)/(max(HX, HY));
    features["IMOC"] = IMOC;

}