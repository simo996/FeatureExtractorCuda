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

double computeASM(const struct GLCM metaGLCM)
{
	double result = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, metaGLCM.maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;
		result += pow((actualPairProbability),2);
	}

	return result;
}

double computeAutocorrelation(const struct GLCM metaGLCM)
{
	double result = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, metaGLCM.maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		result += actualPair.grayLevelI * actualPair. grayLevelJ * actualPairProbability;
	}
	return result;
}


double computeEntropy(const struct GLCM metaGLCM)
{
	double result = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, metaGLCM.maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		result += actualPairProbability * log(actualPairProbability);
		// No pairs with 0 probability, so log is safe
	}

	return (-1 * result);
}

double computeMaximumProbability(const struct GLCM metaGLCM)
{
	double maxProb;
	GrayPair actualPair = unPack(metaGLCM.elements[0], metaGLCM.numberOfPairs, metaGLCM.maxGrayLevel);
	double actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;
	maxProb = actualPairProbability;

	for(int i=1 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, metaGLCM.maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		if(actualPairProbability > maxProb)
		{
			maxProb = actualPairProbability;
		}
	}

	return maxProb;
}

double computeHomogeneity(const struct GLCM metaGLCM)
{
	double result = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, metaGLCM.maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		result += actualPairProbability /
					   (1 + fabs(actualPair.grayLevelI - actualPair.grayLevelJ));

	}

	return result;
}


double computeContrast(const struct GLCM metaGLCM)
{
	double result = 0;
	GrayPair actualPair;
	double actualPairProbability;
	
	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, metaGLCM.maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		result += actualPairProbability
					* (pow(fabs(actualPair.grayLevelI - actualPair.grayLevelJ), 2));
	}

	return result;
}

double computeDissimilarity(const struct GLCM metaGLCM)
{
	double result = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, metaGLCM.maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		result += actualPairProbability *
		(fabs(actualPair.grayLevelI - actualPair.grayLevelJ));
	}

	return result;
}

double computeInverceDifferentMomentNormalized(const struct GLCM metaGLCM)
{
	double result = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, metaGLCM.maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		result += actualPairProbability /
							 ((1+pow((actualPair.grayLevelI - actualPair.grayLevelJ),2))/metaGLCM.maxGrayLevel);
	}

	return result;
}

/* FEATURES WITH MEANS */
double computeCorrelation(const struct GLCM metaGLCM, const double muX, const double muY, const double sigmaX, const double sigmaY)
{
	double result = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, metaGLCM.maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		result += ((actualPair.grayLevelI - muX) * (actualPair.grayLevelJ - muY) * actualPairProbability )
					   /(sigmaX * sigmaY);

	}

	return result;
}

double computeClusterProminecence(const struct GLCM metaGLCM, const double muX, const double muY)
{
	double result = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, metaGLCM.maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		result += pow ((actualPair.grayLevelI + actualPair.grayLevelJ -muX - muY), 4) * actualPairProbability;
	}

	return result;
}

double computeClusterShade(const struct GLCM metaGLCM, const double muX, const double muY)
{
	double result = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, metaGLCM.maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		result += pow ((actualPair.grayLevelI + actualPair.grayLevelJ -muX - muY), 3) * actualPairProbability;
	}

	return result;
}

double computeSumOfSquares(const struct GLCM metaGLCM, const double mu)
{
	double result = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, metaGLCM.maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		result += pow ((actualPair.grayLevelI - mu), 2) * actualPairProbability;
	}

	return result;
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
double computeMean(const struct GLCM metaGLCM)
{
	double mu = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, metaGLCM.maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		mu += actualPairProbability;
	}
	return mu;
}

// Mean of
double computeMuX(const struct GLCM metaGLCM)
{
	double muX = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, metaGLCM.maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		muX += actualPair.grayLevelI * actualPairProbability;
	}
	return muX;
}

double computeMuY(const struct GLCM metaGLCM)
{
	double muY = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, metaGLCM.maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		muY += actualPair.grayLevelJ * actualPairProbability;
	}
	return muY;
}

double computeSigmaX(const struct GLCM metaGLCM, const double muX)
{
	double sigmaX = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, metaGLCM.maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		sigmaX += pow((actualPair.grayLevelI - muX),2) * actualPairProbability;
	}

	return sqrt(sigmaX);
}

double computeSigmaY(const struct GLCM metaGLCM, const double muY)
{
	double sigmaY = 0;
	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i < metaGLCM.numberOfUniquePairs; i++)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, metaGLCM.maxGrayLevel);
		actualPairProbability = ((double) actualPair.multiplicity)/metaGLCM.numberOfPairs;

		sigmaY += pow((actualPair.grayLevelJ - muY),2) * actualPairProbability;
	}

	return sqrt(sigmaY);
}


void computeFeatures(double * output, const struct GLCM metaGLCM)
{
	output[0]= computeASM(metaGLCM);
	output[1]= computeAutocorrelation(metaGLCM);
	output[2]= computeEntropy(metaGLCM);
	output[3]= computeMaximumProbability(metaGLCM);
	output[4]= computeHomogeneity(metaGLCM);
	output[5]= computeContrast(metaGLCM);
	output[6]= computeDissimilarity(metaGLCM);

	double muX, muY, mu, sigmaX, sigmaY;
	mu = computeMean(metaGLCM);
	muX = computeMuX(metaGLCM);
	muY = computeMuY(metaGLCM);
	sigmaX = computeSigmaX(metaGLCM, muX);
	sigmaY = computeSigmaY(metaGLCM, muY);

	output[7]= computeCorrelation(metaGLCM, muX, muY, sigmaX, sigmaY);
	output[8]= computeClusterProminecence(metaGLCM, muX, muY);
	output[9]= computeClusterShade(metaGLCM, muX, muY);
	output[10]= computeSumOfSquares(metaGLCM, mu);
	output[11]= computeInverceDifferentMomentNormalized(metaGLCM);

	int * summedPairs =  (int *) malloc(sizeof(int) * metaGLCM.numberOfUniquePairs);
	int summedPairsLength = codifySummedPairs(metaGLCM, summedPairs);
	output[12]= computeSumAverage(summedPairs, summedPairsLength,  metaGLCM.numberOfPairs);
	output[13]= computeSumEntropy(summedPairs, summedPairsLength,  metaGLCM.numberOfPairs);
	output[14]= computeSumVariance(summedPairs, summedPairsLength, metaGLCM.numberOfPairs, output[12]);
	free(summedPairs);

	int * subtractredPairs = (int *) malloc(sizeof(int) * metaGLCM.numberOfUniquePairs);
	int subtractedPairsLength = codifySubtractedPairs(metaGLCM, subtractredPairs);
	output[15]= computeDifferenceEntropy(subtractredPairs, subtractedPairsLength, metaGLCM.numberOfPairs);
	output[16]= computeDifferenceVariance(subtractredPairs, subtractedPairsLength, metaGLCM.numberOfPairs);
	free(subtractredPairs);
	
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
	cout << "DISSIMILARITY: " << features[6] << endl;

	cout << "\nCORRELATION: " << features[7] << endl;
	cout << "CLUSTER PROMINECENCE: " << features[8] << endl;
	cout << "CLUSTER SHADE: " << features[9] << endl;
	cout << "SUM OF SQUARES: " << features[10] << endl;
	cout << "IDM: " << features[11] << endl;
	
	cout << "\nSUM AVERAGE: " << features[12] << endl;
	cout << "SUM ENTROPY: " << features[13] << endl;
	cout << "SUM VARIANCE: " << features[14] << endl;

	cout << "\nDIFF ENTROPY: " << features[15] << endl;
	cout << "DIFF VARIANCE: " << features[16] << endl;

}
