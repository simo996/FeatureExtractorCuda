//
// Created by simo on 11/07/18.
//

#include <iostream>
#include <cmath>
#include "FeatureComputer.h"

using namespace std;

FeatureComputer::FeatureComputer(const Image& img, const int shiftRows,
                                 const int shiftColumns, const Window& wd): image(img), windowData(wd) {
    windowData.setDirectionShifts(shiftRows, shiftColumns);
}

map<FeatureNames, double> FeatureComputer::computeDirectionalFeatures() {
    GLCM glcm(image, windowData);
    //printGLCM(glcm); Print data and elements for debugging
    map<FeatureNames, double> features = computeBatchFeatures(glcm);
    return features;
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

/* 
    This methods will call the cumulative methods that extract features based
    on their type
*/
map<FeatureNames, double> FeatureComputer::computeBatchFeatures(const GLCM& glcm) {
    map<FeatureNames, double> features;

    // Features computable from glcm Elements
    extractAutonomousFeatures(glcm, features);

    // Feature computable from aggregated glcm pairs
    extractSumAggregatedFeatures(glcm, features);
    extractDiffAggregatedFeatures(glcm, features);

    // Imoc
    extractMarginalFeatures(glcm, features);

    return features;
}

/*
    This method will compute all the features computable from glcm gray level pairs
*/
void FeatureComputer::extractAutonomousFeatures(const GLCM& glcm, map<FeatureNames, double>& features){
    // Intermediate values
    double mean = 0;
    double muX = 0;
    double muY = 0;
    double sigmaX = 0;
    double sigmaY = 0;

    // Actual features
    // TODO think about not retaining local variables and directly accessing the map

    double angularSecondMoment = 0;
    double autoCorrelation = 0;
    double entropy = 0;
    double maxprob = 0;
    double homogeneity = 0;
    double contrast = 0;
    double dissimilarity = 0;
    double idm = 0;

    // First batch of computable features
    typedef map<GrayPair, int>::const_iterator MI;

    for(MI actual=glcm.grayPairsMap.begin() ; actual != glcm.grayPairsMap.end(); actual++)
    {
        GrayPair actualPair = actual->first;
        int i = actualPair.getGrayLevelI();
        int j = actualPair.getGrayLevelJ();
        double actualPairProbability = ((double) actual->second)/glcm.getNumberOfPairs();

        angularSecondMoment += computeAsmStep(actualPairProbability);
        autoCorrelation += computeAutocorrelationStep(i, j, actualPairProbability);
        entropy += computeEntropyStep(actualPairProbability);
        if(maxprob < actualPairProbability)
            maxprob = actualPairProbability;
        homogeneity += computeHomogeneityStep(i, j, actualPairProbability);
        contrast += computeContrastStep(i, j, actualPairProbability);
        dissimilarity += computeDissimilarityStep(i, j, actualPairProbability);
        idm += computeInverceDifferenceMomentStep(i, j, actualPairProbability, glcm.getMaxGrayLevel());

        // intemediate values
        mean += (i * j * actualPairProbability);
        muX += (i * actualPairProbability);
        muY += (j * actualPairProbability);
    }

    features[ASM]= angularSecondMoment;
    features[AUTOCORRELATION]= autoCorrelation;
    features[ENTROPY]= (-1 * entropy);
    features[MAXPROB]= maxprob;
    features[HOMOGENEITY]= homogeneity;
    features[CONTRAST]= contrast;
    features[DISSIMILARITY]= dissimilarity;
    features[IDM]= idm;

    // Second batch of computable features
    double clusterProm = 0;
    double clusterShade = 0;
    double sumOfSquares = 0;

    for(MI actual=glcm.grayPairsMap.begin() ; actual != glcm.grayPairsMap.end(); actual++)
    {
        GrayPair actualPair = actual->first;
        int i = actualPair.getGrayLevelI();
        int j = actualPair.getGrayLevelJ();
        double actualPairProbability = ((double) actual->second)/glcm.getNumberOfPairs();

        clusterProm += computeClusterProminenceStep(i, j, actualPairProbability, muX, muY);
        clusterShade += computeClusterShadeStep(i, j, actualPairProbability, muX, muY);
        sumOfSquares += computeSumOfSquaresStep(i, actualPairProbability, mean);
        sigmaX += pow((i - muX), 2) * actualPairProbability;
        sigmaY+= pow((j - muY), 2) * actualPairProbability;
    }

    sigmaX = sqrt(sigmaX);
    sigmaY = sqrt(sigmaY);

    features[CLUSTERPROMINENCE]= clusterProm;
    features[CLUSTERSHADE]= clusterShade;
    features[SUMOFSQUARES]= sumOfSquares;

    // Only feature that needs the third scan of the glcm
    double correlation = 0;

    for(MI actual=glcm.grayPairsMap.begin() ; actual != glcm.grayPairsMap.end(); actual++)
    {
        GrayPair actualPair = actual->first;
        int i = actualPair.getGrayLevelI();
        int j = actualPair.getGrayLevelJ();
        double actualPairProbability = ((double) actual->second)/glcm.getNumberOfPairs();

        correlation += computeCorrelationStep(i, j, actualPairProbability, 
            muX, muY, sigmaX, sigmaY);
    }
    features[CORRELATION]= correlation;

}

/*
    This method will compute the 3 features obtained from the pairs <k, int freq>
    where k is the sum of the 2 gray leveles <i,j> in a pixel pair of the glcm
*/
void FeatureComputer::extractSumAggregatedFeatures(const GLCM& glcm, map<FeatureNames, double>& features) {
    map<AggregatedGrayPair, int> summedPairs = glcm.codifySummedPairs();
    int numberOfPairs = glcm.getNumberOfPairs();
    // TODO think about not retaining local variables and directly accessing the map
    double sumavg = 0;
    double sumentropy = 0;
    double sumvariance = 0;

    typedef map<AggregatedGrayPair, int>::const_iterator MI;
    for (MI actual = summedPairs.begin(); actual != summedPairs.end(); actual++) 
    {
        AggregatedGrayPair actualPair = actual->first;
        int k = actualPair.getAggregatedGrayLevel();
        double actualPairProbability = ((double) actual->second) / numberOfPairs;

        sumavg += computeSumAverageStep(k, actualPairProbability);
        sumentropy += computeSumEntropyStep(actualPairProbability);
    }
    sumentropy *= -1;
    features[SUMAVERAGE] = sumavg;
    features[SUMENTROPY] = sumentropy;

    for (MI actual = summedPairs.begin(); actual != summedPairs.end(); actual++)
    {
        AggregatedGrayPair actualPair = actual->first;
        int k = actualPair.getAggregatedGrayLevel();
        double actualPairProbability = ((double) actual->second) / numberOfPairs;

        sumvariance += computeSumVarianceStep(k, actualPairProbability, sumentropy);
    }

    features[SUMVARIANCE] = sumvariance;
}

/*
    This method will compute the 2 features obtained from the pairs <k, int freq>
    where k is the absolute difference of the 2 gray leveles in a pixel pair 
    <i,j> of the glcm
*/
void FeatureComputer::extractDiffAggregatedFeatures(const GLCM& glcm, map<FeatureNames, double>& features) {
    map<AggregatedGrayPair, int> subtractedPairs = glcm.codifySubtractedPairs();
    int numberOfPairs= glcm.getNumberOfPairs();
    // TODO think about not retaining local variables and directly accessing the map
    double diffentropy = 0;
    double diffvariance = 0;

    typedef map<AggregatedGrayPair, int>::const_iterator MI;
    for (MI actual = subtractedPairs.begin(); actual != subtractedPairs.end(); actual++) 
    {
        AggregatedGrayPair actualPair = actual->first;
        int k = actualPair.getAggregatedGrayLevel();
        double actualPairProbability = ((double) actual->second) / numberOfPairs;

        diffentropy += computeDiffEntropyStep(actualPairProbability);
        diffvariance += computeDiffVarianceStep(k, actualPairProbability);
    }
    diffentropy *= -1;
    features[DIFFENTROPY] = diffentropy;
    features[DIFFVARIANCE] = diffvariance;

}

/*
    This method will compute the only feature computable from the "marginal 
    representation" of the pairs <(X, ?), int frequency> and the pairs
    <(?, X), int frequency> of reference/neighbor pixel
*/
void FeatureComputer::extractMarginalFeatures(const GLCM& glcm, map<FeatureNames, double>& features){
    map<int, int> marginalPairsX = glcm.codifyXMarginalProbabilities();
    map<int, int> marginalPairsY = glcm.codifyYMarginalProbabilities();
    int numberOfPairs = glcm.getNumberOfPairs();
    double hx = 0;

    // Compute first intermediate value
    typedef map<int, int>::const_iterator MI;
    for (MI actual = marginalPairsX.begin(); actual != marginalPairsX.end(); actual++)
    {
        double probability = ((double) (actual->second)/numberOfPairs);
       
        hx += computeHxStep(probability);
    }
    hx *= -1;

    // Compute second intermediate value
    double hy = 0;
    for (MI actual = marginalPairsY.begin(); actual != marginalPairsY.end(); actual++)
    {

        double probability = ((double) (actual->second)/numberOfPairs);
       
        hy += computeHyStep(probability);
    }
    hy *= -1;

    // Extract third intermediate value
    double hxy = features[ENTROPY];

    // Compute last intermediate value
    double hxy1 = 0;

    typedef map<GrayPair, int>::const_iterator MGPI;
    for (MGPI actual = glcm.grayPairsMap.begin(); actual != glcm.grayPairsMap.end(); actual++)
    {
        GrayPair actualPair = actual->first;
        int i = actualPair.getGrayLevelI();
        int j = actualPair.getGrayLevelJ();
        double actualPairProbability = ((double) actual->second) / numberOfPairs;
        double xMarginalProbability = (double) marginalPairsX.find(i)->second / numberOfPairs;
        double yMarginalProbability = (double) marginalPairsY.find(j)->second / numberOfPairs;

        hxy1 += actualPairProbability * log(xMarginalProbability * yMarginalProbability);
    }
    hxy1 *= -1;
    // TODO think about not retaining local variables and directly accessing the map
    features[IMOC] = (hxy - hxy1)/(max(hx, hy));

}