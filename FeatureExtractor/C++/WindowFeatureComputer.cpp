//
// Created by simo on 11/07/18.
//

#include <iostream>
#include <string>
#include <cmath>
#include "WindowFeatureComputer.h"

using namespace std;

map<string, double> WindowFeatureComputer::computeFeatures(const GLCM& glcm){
    map<string, double> features; // Distrutto all'uscita?
    features["ASM"]= computeASM(glcm);
    features["AUTOCORRELATION"]= computeAutocorrelation(glcm);
    features["ENTROPY"]= computeEntropy(glcm);
    features["MAXPROB"]= computeMaximumProbability(glcm);
    features["HOMOGENEITY"]= computeHomogeneity(glcm);
    features["CONTRAST"]= computeContrast(glcm);
    features["DISSIMILARITY"]= computeDissimilarity(glcm);

    double muX, muY, mu, sigmaX, sigmaY;
    mu = computeMean(glcm);
    muX = computeMuX(glcm);
    muY = computeMuY(glcm);
    sigmaX = computeSigmaX(glcm, muX);
    sigmaY = computeSigmaY(glcm, muY);

    features["CORRELATION"]= computeCorrelation(glcm, muX, muY, sigmaX, sigmaY);
    features["CLUSTER PROMINENCE"]= computeClusterProminence(glcm, muX, muY);
    features["CLUSTER SHADE"]= computeClusterShade(glcm, muX, muY);
    features["SUM OF SQUARES"]= computeSumOfSquares(glcm, mu);
    features["IDM"]= computeInverceDifferentMoment(glcm);

    map<AggregatedGrayPair, int> summedPairs = glcm.codifySummedPairs(); // Distrutto all'uscita?
    int numberOfElements= glcm.getNumberOfPairs();
    features["SUM AVERAGE"]= computeSumAverage(summedPairs,  numberOfElements); // giusto ?
    features["SUM ENTROPY"]= computeSumEntropy(summedPairs,  numberOfElements); // OK
    features["SUM VARIANCE"]= computeSumVariance(summedPairs, features["SUM AVERAGE"], numberOfElements); // giusto ?

    map<AggregatedGrayPair, int> subtractedPairs = glcm.codifySubtractedPairs(); // Distrutto all'uscita?
    features["DIFF ENTROPY"]= computeDifferenceEntropy(subtractedPairs, numberOfElements);
    features["DIFF VARIANCE"]= computeDifferenceVariance(subtractedPairs, numberOfElements);

    return features; // Distrutto all'uscita?
}

void WindowFeatureComputer::printFeatures(map<std::string, double>& features){
    cout << endl;
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
    cout << "SUM OF SQUARES: \t" << features["SUM OF SQUARS"] << endl;
    cout << "IDM normalized: \t" << features["IDM"] << endl;

    cout << "SUM AVERAGE: \t" << features["SUM AVERAGE"] << endl;
    cout << "SUM ENTROPY: \t" << features["SUM ENTROPY"] << endl;
    cout << "SUM VARIANCE: \t" << features["SUM VARIANCE"] << endl;

    cout << "DIFF ENTROPY: \t" << features["DIFF ENTROPY"] << endl;
    cout << "DIFF VARIANCE: \t" << features["DIFF VARIANCE"] << endl;
    cout << endl;

}

double WindowFeatureComputer::computeASM(const GLCM& metaGLCM)
{
    double result = 0;
    double actualPairProbability;

    typedef map<GrayPair, int>::const_iterator MI;
    for(MI actual=metaGLCM.grayPairsMap.begin() ; actual != metaGLCM.grayPairsMap.end(); actual++)
    {
        GrayPair actualPair = actual->first;
        actualPairProbability = ((double) actual->second)/metaGLCM.getNumberOfPairs();

        result += pow((actualPairProbability),2);
    }

    return result;
}

double WindowFeatureComputer::computeAutocorrelation(const GLCM& metaGLCM)
{
    double result = 0;
    double actualPairProbability;

    typedef map<GrayPair, int>::const_iterator MI;
    for(MI actual=metaGLCM.grayPairsMap.begin() ; actual != metaGLCM.grayPairsMap.end(); actual++)
    {
        GrayPair actualPair = actual->first;
        actualPairProbability = ((double) actual->second)/metaGLCM.getNumberOfPairs();

        result += actualPair.getGrayLevelI() * actualPair.getGrayLevelJ() * actualPairProbability;
    }

    return result;
}


double WindowFeatureComputer::computeEntropy(const  GLCM& metaGLCM)
{
    double result = 0;
    double actualPairProbability;

    typedef map<GrayPair, int>::const_iterator MI;
    for(MI actual=metaGLCM.grayPairsMap.begin() ; actual != metaGLCM.grayPairsMap.end(); actual++)
    {
        GrayPair actualPair = actual->first;
        actualPairProbability = ((double) actual->second)/metaGLCM.getNumberOfPairs();

        result += actualPairProbability * log(actualPairProbability);
    }

    return (-1 * result);
}

double WindowFeatureComputer::computeMaximumProbability(const  GLCM& metaGLCM)
{
    double maxProb;
    double actualPairProbability;

    typedef map<GrayPair, int>::const_iterator MI;
    maxProb=metaGLCM.grayPairsMap.begin()->second / metaGLCM.getNumberOfPairs(); // Initialize with first element's frequency

    for(MI actual=metaGLCM.grayPairsMap.begin() ; actual != metaGLCM.grayPairsMap.end(); actual++)
    {
        actualPairProbability = ((double) actual->second)/metaGLCM.getNumberOfPairs();
        if(maxProb < actualPairProbability)
            maxProb = actualPairProbability;
    }

    return maxProb;
}

double WindowFeatureComputer::computeHomogeneity(const  GLCM& metaGLCM)
{
    double result = 0;
    double actualPairProbability;

    typedef map<GrayPair, int>::const_iterator MI;
    for(MI actual=metaGLCM.grayPairsMap.begin() ; actual != metaGLCM.grayPairsMap.end(); actual++)
    {
        GrayPair actualPair = actual->first;
        actualPairProbability = ((double) actual->second)/metaGLCM.getNumberOfPairs();

        result += actualPairProbability / (1 + fabs(actualPair.getGrayLevelI() - actualPair.getGrayLevelJ()));
    }

    return result;
}

double WindowFeatureComputer::computeContrast(const  GLCM& metaGLCM) {
    double result = 0;
    double actualPairProbability;

    typedef map<GrayPair, int>::const_iterator MI;
    for (MI actual = metaGLCM.grayPairsMap.begin(); actual != metaGLCM.grayPairsMap.end(); actual++)
    {
        GrayPair actualPair = actual->first;
        actualPairProbability = ((double) actual->second) / metaGLCM.getNumberOfPairs();

        result += actualPairProbability
                  * (pow(fabs(actualPair.getGrayLevelI() - actualPair.getGrayLevelJ()), 2));
    }

    return result;
}

double WindowFeatureComputer::computeDissimilarity(const  GLCM& metaGLCM)
{
    double result = 0;
    double actualPairProbability;

    typedef map<GrayPair, int>::const_iterator MI;
    for (MI actual = metaGLCM.grayPairsMap.begin(); actual != metaGLCM.grayPairsMap.end(); actual++)
    {
        GrayPair actualPair = actual->first;
        actualPairProbability = ((double) actual->second) / metaGLCM.getNumberOfPairs();

        result += actualPairProbability *
                  (fabs(actualPair.getGrayLevelI() - actualPair.getGrayLevelJ()));
    }

    return result;
}

double WindowFeatureComputer::computeInverceDifferentMoment(const  GLCM& metaGLCM)
{
    double result = 0;
    double actualPairProbability;

    typedef map<GrayPair, int>::const_iterator MI;
    for (MI actual = metaGLCM.grayPairsMap.begin(); actual != metaGLCM.grayPairsMap.end(); actual++) 
    {
        GrayPair actualPair = actual->first;
        actualPairProbability = ((double) actual->second) / metaGLCM.getNumberOfPairs();

        result += actualPairProbability /
                  (1 + fabs(actualPair.getGrayLevelI() - actualPair.getGrayLevelJ())/metaGLCM.getMaxGrayLevel());
    }

    return result;
}

/* FEATURES WITH MEANS */
double WindowFeatureComputer::computeCorrelation(const  GLCM& metaGLCM, const double muX, const double muY, const double sigmaX, const double sigmaY)
{
    double result = 0;
    double actualPairProbability;

    typedef map<GrayPair, int>::const_iterator MI;
    for (MI actual = metaGLCM.grayPairsMap.begin(); actual != metaGLCM.grayPairsMap.end(); actual++)
    {
        GrayPair actualPair = actual->first;
        actualPairProbability = ((double) actual->second) / metaGLCM.getNumberOfPairs();

        result += ((actualPair.getGrayLevelI() - muX) * (actualPair.getGrayLevelJ() - muY) * actualPairProbability )
                  /(sigmaX * sigmaY);
    }

    return result;
}

double WindowFeatureComputer::computeClusterProminence(const  GLCM& metaGLCM, const double muX, const double muY)
{
    double result = 0;
    double actualPairProbability;

    typedef map<GrayPair, int>::const_iterator MI;
    for (MI actual = metaGLCM.grayPairsMap.begin(); actual != metaGLCM.grayPairsMap.end(); actual++)
    {
        GrayPair actualPair = actual->first;
        actualPairProbability = ((double) actual->second) / metaGLCM.getNumberOfPairs();

        result += pow ((actualPair.getGrayLevelI() + actualPair.getGrayLevelJ() -muX - muY), 4) * actualPairProbability;
    }

    return result;
}

double WindowFeatureComputer::computeClusterShade(const  GLCM& metaGLCM, const double muX, const double muY)
{
    double result = 0;
    double actualPairProbability;

    typedef map<GrayPair, int>::const_iterator MI;
    for (MI actual = metaGLCM.grayPairsMap.begin(); actual != metaGLCM.grayPairsMap.end(); actual++) 
    {
        GrayPair actualPair = actual->first;
        actualPairProbability = ((double) actual->second) / metaGLCM.getNumberOfPairs();

        result += pow ((actualPair.getGrayLevelI() + actualPair.getGrayLevelJ() -muX - muY), 3) * actualPairProbability;
    }

    return result;
}

double WindowFeatureComputer::computeSumOfSquares(const  GLCM& metaGLCM, const double mu)
{
    double result = 0;
    double actualPairProbability;

    typedef map<GrayPair, int>::const_iterator MI;
    for (MI actual = metaGLCM.grayPairsMap.begin(); actual != metaGLCM.grayPairsMap.end(); actual++) 
    {
        GrayPair actualPair = actual->first;
        actualPairProbability = ((double) actual->second) / metaGLCM.getNumberOfPairs();

        result += pow ((actualPair.getGrayLevelI() - mu), 2) * actualPairProbability;
    }

    return result;
}

// SUM Aggregated features
double WindowFeatureComputer::computeSumAverage(const map<AggregatedGrayPair, int>& summedMetaGLCM, const int numberOfPairs)
{
    double result = 0;
    double actualPairProbability;

    typedef map<AggregatedGrayPair, int>::const_iterator MI;
    for (MI actual = summedMetaGLCM.begin(); actual != summedMetaGLCM.end(); actual++) 
    {
        AggregatedGrayPair actualPair = actual->first;
        actualPairProbability = ((double) actual->second) / numberOfPairs;

        result += actualPair.getAggregatedGrayLevel() * actualPairProbability;
    }

    return result;
}


double WindowFeatureComputer::computeSumEntropy(const map<AggregatedGrayPair, int>& summedMetaGLCM, const int numberOfPairs)
{
    double result = 0;
    double actualPairProbability;

    typedef map<AggregatedGrayPair, int>::const_iterator MI;
    for (MI actual = summedMetaGLCM.begin(); actual != summedMetaGLCM.end(); actual++) 
    {
        actualPairProbability = ((double) actual->second) / numberOfPairs;

        result += log(actualPairProbability) * actualPairProbability;
    }

    return -1 * result;
}

double WindowFeatureComputer::computeSumVariance(const map<AggregatedGrayPair, int>& summedMetaGLCM, const double sumEntropy, const int numberOfPairs)
{
    double result = 0;
    double actualPairProbability;

    typedef map<AggregatedGrayPair, int>::const_iterator MI;
    for (MI actual = summedMetaGLCM.begin(); actual != summedMetaGLCM.end(); actual++) 
    {
        AggregatedGrayPair actualPair = actual->first;
        actualPairProbability = ((double) actual->second) / numberOfPairs;

        result += pow((actualPair.getAggregatedGrayLevel() - sumEntropy),2)
                  * actualPairProbability;
    }

    return result;
}

// DIFFERENCE
double WindowFeatureComputer::computeDifferenceEntropy(const map<AggregatedGrayPair, int>& subtractedMetaGLCM, const int numberOfPairs)
{
    double result = 0;
    double actualPairProbability;

    typedef map<AggregatedGrayPair, int>::const_iterator MI;
    for (MI actual = subtractedMetaGLCM.begin(); actual != subtractedMetaGLCM.end(); actual++)
    {
        actualPairProbability = ((double) actual->second) / numberOfPairs;

        result += log(actualPairProbability) * actualPairProbability;
    }
    return -1*result;
}

double WindowFeatureComputer::computeDifferenceVariance(const map<AggregatedGrayPair, int>& subtractedMetaGLCM, const int numberOfPairs)
{
    double result = 0;
    double actualPairProbability;

    typedef map<AggregatedGrayPair, int>::const_iterator MI;
    for (MI actual = subtractedMetaGLCM.begin(); actual != subtractedMetaGLCM.end(); actual++)
    {
        AggregatedGrayPair actualPair = actual->first;
        actualPairProbability = ((double) actual->second) / numberOfPairs;

        result += pow(actualPair.getAggregatedGrayLevel(), 2) * actualPairProbability;
    }

    return result;
}


// Mean of all probabilities
// Same implementation of Autocorrelation
double WindowFeatureComputer::computeMean(const  GLCM& metaGLCM)
{
    double mu = 0;
    double actualPairProbability;

    typedef map<GrayPair, int>::const_iterator MI;
    for (MI actual = metaGLCM.grayPairsMap.begin(); actual != metaGLCM.grayPairsMap.end(); actual++) 
    {
        GrayPair actualPair = actual->first;
        actualPairProbability = ((double) actual->second) / metaGLCM.getNumberOfPairs();

        mu += (actualPair.getGrayLevelI()) * (actualPair.getGrayLevelJ()) * actualPairProbability;
    }
    
    return mu;
}


// Mean of (i,*)
double WindowFeatureComputer::computeMuX(const GLCM& metaGLCM)
{
    double muX = 0;
    double actualPairProbability;

    typedef map<GrayPair, int>::const_iterator MI;
    for (MI actual = metaGLCM.grayPairsMap.begin(); actual != metaGLCM.grayPairsMap.end(); actual++) 
    {
        GrayPair actualPair = actual->first;
        actualPairProbability = ((double) actual->second) / metaGLCM.getNumberOfPairs();

        muX += actualPair.getGrayLevelI() * actualPairProbability;
    }
    return muX;
}

// Mean of (*,i)
double WindowFeatureComputer::computeMuY(const GLCM& metaGLCM)
{
    double muY = 0;
    double actualPairProbability;

    typedef map<GrayPair, int>::const_iterator MI;
    for (MI actual = metaGLCM.grayPairsMap.begin(); actual != metaGLCM.grayPairsMap.end(); actual++) 
    {
        GrayPair actualPair = actual->first;
        actualPairProbability = ((double) actual->second) / metaGLCM.getNumberOfPairs();

        muY += actualPair.getGrayLevelJ() * actualPairProbability;
    }
    return muY;
}

// Variance of (i,*)
double WindowFeatureComputer::computeSigmaX(const GLCM& metaGLCM, const double muX)
{
    double sigmaX = 0;
    double actualPairProbability;

    typedef map<GrayPair, int>::const_iterator MI;
    for (MI actual = metaGLCM.grayPairsMap.begin(); actual != metaGLCM.grayPairsMap.end(); actual++) 
    {
        GrayPair actualPair = actual->first;
        actualPairProbability = ((double) actual->second) / metaGLCM.getNumberOfPairs();

        sigmaX += pow((actualPair.getGrayLevelI() - muX),2) * actualPairProbability;
    }

    return sqrt(sigmaX);
}

// Variance of (*,i)
double WindowFeatureComputer::computeSigmaY(const GLCM& metaGLCM, const double muY)
{
    double sigmaY = 0;
    double actualPairProbability;

    typedef map<GrayPair, int>::const_iterator MI;
    for (MI actual = metaGLCM.grayPairsMap.begin(); actual != metaGLCM.grayPairsMap.end(); actual++) 
    {
        GrayPair actualPair = actual->first;
        actualPairProbability = ((double) actual->second) / metaGLCM.getNumberOfPairs();

        sigmaY+= pow((actualPair.getGrayLevelJ() - muY),2) * actualPairProbability;
    }

    return sqrt(sigmaY);
}