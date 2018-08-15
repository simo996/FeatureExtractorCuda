//
// Created by simo on 11/07/18.
//

#include <iostream>
#include <algorithm>
#include <cmath>
#include "FeatureComputer.h"

using namespace std;

FeatureComputer::FeatureComputer(const unsigned int * pixels, const ImageData& img, const int shiftRows,
                                 const int shiftColumns, const Window& wd, const WorkArea wa)
                                 : pixels(pixels), image(img), windowData(wd) , workArea(wa) {
    windowData.setDirectionShifts(shiftRows, shiftColumns);
}

vector<double> FeatureComputer::computeDirectionalFeatures() {
    GLCM glcm(pixels, image, windowData, workArea);
    //printGLCM(glcm); // Print data and elements for debugging
    vector<double> features = computeBatchFeatures(glcm);
    return features;
}

/* TODO remove METHODS FOR DEBUG */
void FeatureComputer::printGLCM(const GLCM& glcm){
    glcm.printGLCMData();
    glcm.printGLCMElements();
    glcm.printAggregated();
}

// ASM
inline double computeAsmStep(const double actualPairProbability){
    return pow((actualPairProbability),2);
}

// AUTOCORRELATION
inline double computeAutocorrelationStep(const uint i, const uint j, const double actualPairProbability){
    return (i * j * actualPairProbability);
}

// ENTROPY
inline double computeEntropyStep(const double actualPairProbability){
    return (actualPairProbability * log(actualPairProbability));
}

// HOMOGENEITY
inline double computeHomogeneityStep(const uint i, const uint j, const double actualPairProbability){
    int diff = i - j; // avoids casting value errors of uint(negative number)
    return (actualPairProbability / (1 + fabs(diff)));
}

// CONTRAST
inline double computeContrastStep(const uint i, const uint j, const double actualPairProbability){
    int diff = i - j; // avoids casting value errors of uint(negative number)
    return (actualPairProbability * (pow(fabs(diff), 2)));
}

// DISSIMILARITY
inline double computeDissimilarityStep(const uint i, const uint j, const double pairProbability){
    int diff = i - j; // avoids casting value errors of uint(negative number)
    return (pairProbability * (fabs(diff)));
}

// IDM
inline double computeInverceDifferenceMomentStep(const uint i, const uint j,
    const double pairProbability, const uint maxGrayLevel)
{
    int diff = i - j; // avoids casting value errors of uint(negative number)
    return (pairProbability / (1 + fabs(diff) / maxGrayLevel));
}

/* FEATURES WITH MEANS */
// CORRELATION
inline double computeCorrelationStep(const uint i, const uint j, 
    const double pairProbability, const double muX, const double muY, 
    const double sigmaX, const double sigmaY)
{
    // beware ! unsigned int - double
    return (((i - muX) * (j - muY) * pairProbability ) / (sigmaX * sigmaY));
}

// CLUSTER PROMINENCE
inline double computeClusterProminenceStep(const uint i, const uint j, 
    const double pairProbability, const double muX, const double muY){
    return (pow((i + j - muX - muY), 4) * pairProbability);
}

// CLUSTER SHADE
inline double computeClusterShadeStep(const uint i, const uint j,
    const double pairProbability, const double muX, const double muY){
    return (pow((i + j - muX - muY), 3) * pairProbability);
}

// SUM OF SQUARES
inline double computeSumOfSquaresStep(const uint i,
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
inline double computeSumVarianceStep(const uint aggregatedGrayLevel, 
    const double pairProbability, const double sumEntropy){
    // beware ! unsigned int - double
    return (pow((aggregatedGrayLevel - sumEntropy),2) * pairProbability);
}

// DIFF Aggregated features
// DIFF ENTROPY
inline double computeDiffEntropyStep(const double pairProbability){
    return (log(pairProbability) * pairProbability);
}

// DIFF
inline double computeDiffVarianceStep(const uint aggregatedGrayLevel, const double pairProbability){
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
vector<double> FeatureComputer::computeBatchFeatures(const GLCM& glcm) {
    vector<double> features(Features::getAllSupportedFeatures().size());

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
void FeatureComputer::extractAutonomousFeatures(const GLCM& glcm, vector<double>& features){
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
    int length = glcm.getNumberOfUniquePairs();
    for (int k = 0; k < length; ++k) {
        GrayPair actualPair = glcm.elements[k];

        uint i = actualPair.getGrayLevelI();
        uint j = actualPair.getGrayLevelJ();
        double actualPairProbability = ((double) actualPair.getFrequency())/glcm.getNumberOfPairs();

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

    for (int k = 0; k < length; ++k)
    {
        GrayPair actualPair = glcm.elements[k];
        uint i = actualPair.getGrayLevelI();
        uint j = actualPair.getGrayLevelJ();
        double actualPairProbability = ((double) actualPair.getFrequency())/glcm.getNumberOfPairs();

        clusterProm += computeClusterProminenceStep(i, j, actualPairProbability, muX, muY);
        clusterShade += computeClusterShadeStep(i, j, actualPairProbability, muX, muY);
        sumOfSquares += computeSumOfSquaresStep(i, actualPairProbability, mean);
        sigmaX += pow((i - muX), 2) * actualPairProbability;
        sigmaY += pow((j - muY), 2) * actualPairProbability;
    }

    sigmaX = sqrt(sigmaX);
    sigmaY = sqrt(sigmaY);

    features[CLUSTERPROMINENCE]= clusterProm;
    features[CLUSTERSHADE]= clusterShade;
    features[SUMOFSQUARES]= sumOfSquares;

    // Only feature that needs the third scan of the glcm
    double correlation = 0;

    for (int k = 0; k < length; ++k)
    {
        GrayPair actualPair = glcm.elements[k];
        uint i = actualPair.getGrayLevelI();
        uint j = actualPair.getGrayLevelJ();
        double actualPairProbability = ((double) actualPair.getFrequency())/glcm.getNumberOfPairs();

        correlation += computeCorrelationStep(i, j, actualPairProbability, 
            muX, muY, sigmaX, sigmaY);
    }
    features[CORRELATION]= correlation;

}

/*
    This method will compute the 3 features obtained from the pairs <k, int freq>
    where k is the sum of the 2 gray leveles <i,j> in a pixel pair of the glcm
*/
void FeatureComputer::extractSumAggregatedFeatures(const GLCM& glcm, vector<double>& features) {
    int numberOfPairs = glcm.getNumberOfPairs();
    // TODO think about not retaining local variables and directly accessing the map
    double sumavg = 0;
    double sumentropy = 0;
    double sumvariance = 0;

    // First batch of computable features
    int length = glcm.getNumberOfUniqueAggregatedElements(glcm.summedPairs);
    for (int i = 0; i < length; ++i) {
        AggregatedGrayPair actualPair = glcm.summedPairs[i];
        uint k = actualPair.getAggregatedGrayLevel();
        double actualPairProbability = ((double) actualPair.getFrequency()) / numberOfPairs;

        sumavg += computeSumAverageStep(k, actualPairProbability);
        sumentropy += computeSumEntropyStep(actualPairProbability);
    }
    sumentropy *= -1;
    features[SUMAVERAGE] = sumavg;
    features[SUMENTROPY] = sumentropy;

    for (int i = 0; i < length; ++i) {
        AggregatedGrayPair actualPair = glcm.summedPairs[i];
        uint k = actualPair.getAggregatedGrayLevel();
        double actualPairProbability = ((double) actualPair.getFrequency()) / numberOfPairs;

        sumvariance += computeSumVarianceStep(k, actualPairProbability, sumentropy);
    }

    features[SUMVARIANCE] = sumvariance;
}

/*
    This method will compute the 2 features obtained from the pairs <k, int freq>
    where k is the absolute difference of the 2 gray leveles in a pixel pair 
    <i,j> of the glcm
*/
void FeatureComputer::extractDiffAggregatedFeatures(const GLCM& glcm, vector<double>& features) {
    int numberOfPairs= glcm.getNumberOfPairs();
    // TODO think about not retaining local variables and directly accessing the map
    double diffentropy = 0;
    double diffvariance = 0;

    int length = glcm.getNumberOfUniqueAggregatedElements(glcm.subtractedPairs);
    for (int i = 0; i < length; ++i) {
        AggregatedGrayPair actualPair = glcm.subtractedPairs[i];
        uint k = actualPair.getAggregatedGrayLevel();
        double actualPairProbability = ((double) actualPair.getFrequency()) / numberOfPairs;

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
void FeatureComputer::extractMarginalFeatures(const GLCM& glcm, vector<double>& features){
    int numberOfPairs = glcm.getNumberOfPairs();
    double hx = 0;

    // Compute first intermediate value
    int xLength = glcm.getNumberOfUniqueAggregatedElements(glcm.xMarginalPairs);
    for (int k = 0; k < xLength; ++k) {
        double probability = ((double) (glcm.xMarginalPairs[k].getFrequency())/numberOfPairs);

        hx += computeHxStep(probability);
    }
    hx *= -1;

    // Compute second intermediate value
    double hy = 0;
    int yLength = glcm.getNumberOfUniqueAggregatedElements(glcm.yMarginalPairs);
    for (int k = 0; k < yLength; ++k) {
        double probability = ((double) (glcm.yMarginalPairs[k].getFrequency())/numberOfPairs);
       
        hy += computeHyStep(probability);
    }
    hy *= -1;

    // Extract third intermediate value
    double hxy = features[ENTROPY];

    // Compute last intermediate value
    double hxy1 = 0;

    int length = glcm.getNumberOfUniquePairs();
    for (int l = 0; l < length; ++l) {
        GrayPair actualPair = glcm.elements[l];
        double actualPairProbability = ((double) glcm.elements[l].getFrequency()) / numberOfPairs;

        AggregatedGrayPair i (actualPair.getGrayLevelI(), 0); // 0 frequency is placeholder
        auto xposition = find(glcm.xMarginalPairs.begin(), glcm.xMarginalPairs.end(), i);
        double xMarginalProbability = (double) xposition.operator*().getFrequency() / numberOfPairs;

        AggregatedGrayPair j (actualPair.getGrayLevelJ(), 0); // 0 frequency is placeholder
        auto yposition = find(glcm.yMarginalPairs.begin(), glcm.yMarginalPairs.end(), j);
        double yMarginalProbability = (double) yposition.operator*().getFrequency() / numberOfPairs;

        hxy1 += actualPairProbability * log(xMarginalProbability * yMarginalProbability);
    }
    hxy1 *= -1;
    // TODO think about not retaining local variables and directly accessing the map
    features[IMOC] = (hxy - hxy1)/(max(hx, hy));

}