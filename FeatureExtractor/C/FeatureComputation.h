//
// Created by simo on 07/05/18.
//

#ifndef FEATURESEXTRACTOR_FEATURECOMPUTATION_H
#define FEATURESEXTRACTOR_FEATURECOMPUTATION_H

double computeASM(const struct GLCM metaGLCM, const int maxGrayLevel);
double computeAutocorrelation(const struct GLCM metaGLCM, const int maxGrayLevel);
double computeEntropy(const struct GLCM metaGLCM, const int maxGrayLevel);
double computeMaximumProbability(const struct GLCM metaGLCM, const int maxGrayLevel);
double computeHomogeneity(const struct GLCM metaGLCM, const int maxGrayLevel);
double computeContrast(const struct GLCM metaGLCM, const int maxGrayLevel);
double computeInverceDifferentMomentNormalized(const struct GLCM metaGLCM, const int maxGrayLevel);

double computeCorrelation(const struct GLCM metaGLCM, const int maxGrayLevel, const double muX, const double muY, const double sigmaX, const double sigmaY);
double computeClusterProminecence(const struct GLCM metaGLCM, const int maxGrayLevel, const double muX, const double muY);
double computeClusterShade(const struct GLCM metaGLCM, const int maxGrayLevel, const double muX, const double muY);
double computeSumOfSquares(const struct GLCM metaGLCM, const int maxGrayLevel, const double mu);

double computeSumAverage(const int * summedMetaGLCM, const int length, const int numberOfPairs);
double computeSumEntropy(const int * summedMetaGLCM, const int length, const int numberOfPairs);
double computeSumVariance(const int * summedMetaGLCM, const int length, const int numberOfPairs, const double sumEntropy);

double computeDifferenceEntropy(const int * aggregatedMetaGLCM, const int length, const int numberOfPairs);
double computeDifferenceVariance(const int * aggregatedMetaGLCM, const int length, const int numberOfPairs);

double computeMean(const struct GLCM metaGLCM, const int maxGrayLevel);
double computeMuX(const struct GLCM metaGLCM, const int maxGrayLevel);
double computeMuY(const struct GLCM metaGLCM, const int maxGrayLevel);

double computeSigmaX(const struct GLCM metaGLCM, const int maxGrayLevel, const double muX);
double computeSigmaY(const struct GLCM metaGLCM, const int maxGrayLevel, const double muY);

void computeFeatures(double * output, const struct GLCM metaGLCM, const int maxGrayLevel);
void printFeatures(double * features);


#endif //FEATURESEXTRACTOR_FEATURECOMPUTATION_H
