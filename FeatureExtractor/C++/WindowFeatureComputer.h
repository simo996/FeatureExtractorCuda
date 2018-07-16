//
// Created by simo on 11/07/18.
//

#ifndef FEATUREEXTRACTOR_FEATURECOMPUTER_H
#define FEATUREEXTRACTOR_FEATURECOMPUTER_H


#include <vector>
#include "GLCM.h"

enum FeatureList { ASM, AUTOCCORELATION, ENTROPY, MAXPROB, HOMOGENEITY, CONTRAST, DISSIMILARITY, IDM, CORRELATION,
        CLUSTERPROMINENCE, CLUSTERSHADE, SUMOFSQUARES, SUMAVERAGE, SUMENTROPY, SUMVARIANCE, DIFFENTROPY, DIFFVARIANCE
};


class WindowFeatureComputer {
public:
    /*
     * RESPONSABILITA CLASSE: Data la window, generare 1a alla volta le 4 glcm e farci computare le features
     */
    map<std::string, double> computeFeatures(const GLCM& glcm);
    void printFeatures(map<std::string, double>& features);

private:
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
};


#endif //FEATUREEXTRACTOR_FEATURECOMPUTER_H
