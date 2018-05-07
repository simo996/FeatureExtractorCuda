//
// Created by simo on 07/05/18.
//

#ifndef FEATURESEXTRACTOR_FEATURECOMPUTATION_H
#define FEATURESEXTRACTOR_FEATURECOMPUTATION_H

double computeASM(const int * metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel);
double computeAutocorrelation(const int * metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel);
double computeEntropy(const int * metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel);
double computeMaximumProbability(const int * metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel);
double computeHomogeneity(const int * metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel);
double computeContrast(const int * metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel);

double computeCorrelation(const int * metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel, const double muX, const double muY, const double sigmaX, const double sigmaY);
double computeClusterProminecence(const int * metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel, const double muX, const double muY);
double computeClusterShade(const int * metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel, const double muX, const double muY);
double computeSumOfSquares(const int * metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel, const double mu);
double computeInverceDifferentMomentNormalized(const int * metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel);

double computeSumAverage(const int * summedMetaGLCM, const int length, const int numberOfPairs);
double computeSumEntropy(const int * summedMetaGLCM, const int length, const int numberOfPairs);
double computeSumVariance(const int * summedMetaGLCM, const int length, const int numberOfPairs, const double sumEntropy);

double computeDifferenceEntropy(const int * aggregatedMetaGLCM, const int length, const int numberOfPairs);
double computeDifferenceVariance(const int * aggregatedMetaGLCM, const int length, const int numberOfPairs);

double computeMean(const int* metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel);
double computeMuX(const int* metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel);
double computeMuY(const int* metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel);

double computeSigmaX(const int* metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel, const double muX);
double computeSigmaY(const int* metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel, const double muY);

void computeFeatures(double * output, const int * metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel);
void printFeatures(double * features);


#endif //FEATURESEXTRACTOR_FEATURECOMPUTATION_H
