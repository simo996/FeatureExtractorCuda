//
// Created by simo on 07/05/18.
//

#include "FeatureComputation.h"
#include "GrayPair.h"
#include "MetaGLCM.h"
#include "SupportCode.h"

#include <math.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

double computeASM(const struct GLCM metaGLCM, const int maxGrayLevel)
{
	double angularSecondMoment = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;
		angularSecondMoment += pow((actualPairProbability),2);
	}

	return angularSecondMoment;
}

double computeAutocorrelation(const struct GLCM metaGLCM, const int maxGrayLevel)
{
	double autocorrelation = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		autocorrelation += actualPair.grayLevelI * actualPair. grayLevelJ * actualPairProbability;
	}
	return autocorrelation;
}


double computeEntropy(const struct GLCM metaGLCM, const int maxGrayLevel)
{
	double entropy = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		entropy += actualPairProbability * log(actualPairProbability);
		// No pairs with 0 probability, so log is safe
	}

	return (-1*entropy);
}

double computeMaximumProbability(const struct GLCM metaGLCM, const int maxGrayLevel)
{
	double maxProb;
	GrayPair actualPair = unPack(metaGLCM.elements[0], metaGLCM.numberOfPairs, maxGrayLevel);
	double actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;
	maxProb = actualPairProbability;

	for(int i=1 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		if(actualPairProbability > maxProb)
		{
			maxProb = actualPairProbability;
		}
	}

	return maxProb;
}

double computeHomogeneity(const struct GLCM metaGLCM, const int maxGrayLevel)
{
	double homogeneity = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		homogeneity += actualPairProbability /
					   (1 + fabs(actualPair.grayLevelI - actualPair.grayLevelJ));

	}

	return homogeneity;
}


double computeContrast(const struct GLCM metaGLCM, const int maxGrayLevel)
{
	double contrast = 0;
	GrayPair actualPair;
	double actualPairProbability;
	printGLCMData(metaGLCM);
	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		contrast += actualPairProbability
					* (pow(fabs(actualPair.grayLevelI - actualPair.grayLevelJ), 2));
	}

	return contrast;
}

double computeInverceDifferentMomentNormalized(const struct GLCM metaGLCM, const int maxGrayLevel)
{
	double inverceDifference = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		inverceDifference += actualPairProbability /
							 ((1+pow((actualPair.grayLevelI - actualPair.grayLevelJ),2))/maxGrayLevel);
	}

	return inverceDifference;
}

/* FEATURES WITH MEANS */
double computeCorrelation(const struct GLCM metaGLCM, const int maxGrayLevel, const double muX, const double muY, const double sigmaX, const double sigmaY)
{
	double correlation = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		correlation += ((actualPair.grayLevelI - muX) * (actualPair.grayLevelJ - muY) * actualPairProbability )
					   /(sigmaX * sigmaY);

	}

	return correlation;
}

double computeClusterProminecence(const struct GLCM metaGLCM, const int maxGrayLevel, const double muX, const double muY)
{
	double clusterProminecence = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		clusterProminecence += pow ((actualPair.grayLevelI + actualPair.grayLevelJ -muX - muY), 4) * actualPairProbability;
	}

	return clusterProminecence;
}

double computeClusterShade(const struct GLCM metaGLCM, const int maxGrayLevel, const double muX, const double muY)
{
	double clusterShade = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		clusterShade += pow ((actualPair.grayLevelI + actualPair.grayLevelJ -muX - muY), 3) * actualPairProbability;
	}

	return clusterShade;
}

double computeSumOfSquares(const struct GLCM metaGLCM, const int maxGrayLevel, const double mu)
{
	double sumSquares = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		sumSquares += pow ((actualPair.grayLevelI - mu), 2) * actualPairProbability;
	}

	return sumSquares;
}



// SUM Aggregated features
double computeSumAverage(const int * summedMetaGLCM, const int length, const int numberOfPairs)
{
	double result = 0;
	AggregatedPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < length; i++)
	{
		actualPair = aggregatedPairUnPack(summedMetaGLCM[i], numberOfPairs);
		actualPairProbability = ((double) actualPair.multiplicity)/numberOfPairs;
		result += actualPair.aggregatedGrayLevels * actualPairProbability;
	}
	return result;
}

double computeSumEntropy(const int * summedMetaGLCM, const int length, const int numberOfPairs)
{
	double result = 0;
	AggregatedPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < length;  i++)
	{
		actualPair = aggregatedPairUnPack(summedMetaGLCM[i], numberOfPairs);
		actualPairProbability = ((double) actualPair.multiplicity)/numberOfPairs;
		
		result += log(actualPairProbability) * actualPairProbability;
	}
	return -1 * result;
}

double computeSumVariance(const int * summedMetaGLCM, const int length, const int numberOfPairs, const double sumEntropy)
{
	double result = 0;
	AggregatedPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < length ; i++)
	{
		actualPair = aggregatedPairUnPack(summedMetaGLCM[i], numberOfPairs);
		actualPairProbability = ((double) actualPair.multiplicity)/numberOfPairs;

		result += pow((actualPair.aggregatedGrayLevels - sumEntropy),2)
				  * actualPairProbability;
	}
	return result;
}

// DIFFERENCE 
double computeDifferenceEntropy(const int * aggregatedMetaGLCM, const int length, const int numberOfPairs)
{
	double result = 0;
	AggregatedPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < length; i++)
	{
		actualPair = aggregatedPairUnPack(aggregatedMetaGLCM[i], numberOfPairs);
		actualPairProbability = ((double) actualPair.multiplicity)/numberOfPairs;

		result += log(actualPairProbability) * actualPairProbability;
	}
	return -1*result;
}

double computeDifferenceVariance(const int * aggregatedMetaGLCM, const int length, const int numberOfPairs)
{
	double result = 0;
	AggregatedPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < length; i++)
	{
		actualPair = aggregatedPairUnPack(aggregatedMetaGLCM[i], numberOfPairs);
		actualPairProbability = ((double) actualPair.multiplicity)/numberOfPairs;

		result += pow(actualPair.aggregatedGrayLevels, 2) * actualPairProbability;
	}
	return result;
}


// Mean of all probabilities
double computeMean(const struct GLCM metaGLCM, const int maxGrayLevel)
{
	double mu = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		mu += actualPairProbability;
	}
	return mu;
}

// Mean of
double computeMuX(const struct GLCM metaGLCM, const int maxGrayLevel)
{
	double muX = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		muX += actualPair.grayLevelI * actualPairProbability;
	}
	return muX;
}

double computeMuY(const struct GLCM metaGLCM, const int maxGrayLevel)
{
	double muY = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		muY += actualPair.grayLevelJ * actualPairProbability;
	}
	return muY;
}

double computeSigmaX(const struct GLCM metaGLCM, const int maxGrayLevel, const double muX)
{
	double sigmaX = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		sigmaX += pow((actualPair.grayLevelI - muX),2) * actualPairProbability;
	}

	return sqrt(sigmaX);
}

double computeSigmaY(const struct GLCM metaGLCM, const int maxGrayLevel, const double muY)
{
	double sigmaY = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		sigmaY += pow((actualPair.grayLevelJ - muY),2) * actualPairProbability;
	}

	return sqrt(sigmaY);
}


void computeFeatures(double * output, const struct GLCM metaGLCM, const int maxGrayLevel)
{
	output[0]= computeASM(metaGLCM, maxGrayLevel);
	output[1]= computeAutocorrelation(metaGLCM, maxGrayLevel);
	output[2]= computeEntropy(metaGLCM, maxGrayLevel);
	output[3]= computeMaximumProbability(metaGLCM,maxGrayLevel);
	output[4]= computeHomogeneity(metaGLCM, maxGrayLevel);
	output[5]= computeContrast(metaGLCM, maxGrayLevel);

	double muX, muY, mu, sigmaX, sigmaY;
	mu = computeMean(metaGLCM, maxGrayLevel);
	muX = computeMuX(metaGLCM, maxGrayLevel);
	muY = computeMuY(metaGLCM, maxGrayLevel);
	sigmaX = computeSigmaX(metaGLCM, maxGrayLevel, muX);
	sigmaY = computeSigmaY(metaGLCM, maxGrayLevel, muY);

	output[6]= computeCorrelation(metaGLCM, maxGrayLevel, muX, muY, sigmaX, sigmaY);
	output[7]= computeClusterProminecence(metaGLCM, maxGrayLevel, muX, muY);
	output[8]= computeClusterShade(metaGLCM, maxGrayLevel, muX, muY);
	output[9]= computeSumOfSquares(metaGLCM, maxGrayLevel, mu);
	output[10]= computeInverceDifferentMomentNormalized(metaGLCM, maxGrayLevel);

	int * summedPairs =  (int *) malloc(sizeof(int) * metaGLCM.numberOfUniquePairs);
	int summedPairsLength = codifySummedPairs(metaGLCM, summedPairs, maxGrayLevel);
	output[11]= computeSumAverage(summedPairs, summedPairsLength,  metaGLCM.numberOfPairs);
	output[12]= computeSumEntropy(summedPairs, summedPairsLength,  metaGLCM.numberOfPairs);
	output[13]= computeSumVariance(summedPairs, summedPairsLength, metaGLCM.numberOfPairs, output[12]);
	free(summedPairs);

	/*
	int * subtractredPairs = (int *) malloc(sizeof(int) * metaGLCM.numberOfUniquePairs);
	int subtractedPairsLength = codifySubtractedPairs(metaGLCM, subtractredPairs, maxGrayLevel);
	for(int i=0; i < subtractedPairsLength; i++)
	{
		printAggregatedPair(subtractredPairs[i], metaGLCM.numberOfPairs);
	}
	output[14]= computeDifferenceEntropy(subtractredPairs, subtractedPairsLength, metaGLCM.numberOfPairs);
	output[15]= computeDifferenceVariance(subtractredPairs, subtractedPairsLength, metaGLCM.numberOfPairs);
	free(subtractredPairs);
	*/
	
}


void printFeatures(double * features)
{
	cout << endl;
	cout << "ASM: " << features[0] << endl;
	cout << "AUTOCORRELATION: " << features[1] << endl;
	cout << "ENTROPY: " << features[2] << endl;
	cout << "MAXIMUM PROBABILITY: " << features[3] << endl;
	cout << "HOMOGENEITY: " << features[4] << endl;
	cout << "CONTRAST: " << features[5] << endl;

	cout << "CORRELATION: " << features[6] << endl;
	cout << "CLUSTER PROMINECENCE: " << features[7] << endl;
	cout << "CLUSTER SHADE: " << features[8] << endl;
	cout << "SUM OF SQUARES: " << features[9] << endl;
	cout << "IDM: " << features[10] << endl;

	
	cout << "SUM AVERAGE: " << features[11] << endl;
	cout << "SUM ENTROPY: " << features[12] << endl;
	cout << "SUM VARIANCE: " << features[13] << endl;
	/*
	cout << "DIFF ENTROPY: " << features[14] << endl;
	cout << "DIFF VARIANCE: " << features[15] << endl;
	*/
}
