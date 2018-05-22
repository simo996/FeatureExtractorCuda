//
// Created by simo on 07/05/18.
//

#include <iostream>
#include <climits>
#include <stdlib.h>
#include <cstdio>
#include <cstring>

#include "MetaGLCM.h"
#include "SupportCode.h"
#include "GrayPair.h"


void printGLCMData(const GLCM input)
{
	std::cout << std::endl;
	std::cout << "Shift X : " << input.shiftX << std::endl;
	std::cout << "Shift Y: " << input.shiftY  << std::endl;
	std::cout << "Father Window dimension: "<< input.windowDimension  << std::endl;
	std::cout << "Border X: "<< input.borderX  << std::endl;
	std::cout << "Border Y:" << input.borderY  << std::endl;
	std::cout << "Number of Elements: " << input.numberOfPairs  << std::endl;
	std::cout << "Number of unique elements: " << input.numberOfUniquePairs  << std::endl;
	std::cout << std::endl;
}

void printGLCMData(struct GLCM * input)
{
	printGLCMData(*input);
}


void printMetaGlcm(const struct GLCM metaGLCM)
{
	std::cout << std::endl;
	for (int i = 0; i < metaGLCM.numberOfUniquePairs; ++i)
	{
		printPair(metaGLCM.elements[i], metaGLCM.numberOfPairs, metaGLCM.maxGrayLevel);
	}
	std::cout << std::endl;

}

void printAggrregatedMetaGlcm(const int * aggregatedList, const int length, const int numberOfUniquePairs)
{
	std::cout << std::endl;
	for (int i = 0; i < length; ++i)
	{
		printAggregatedPair(aggregatedList[i], numberOfUniquePairs);
	}
	std::cout << std::endl;

}

// Initialized MetaData of the GLCM
struct GLCM initializeMetaGLCM(const int distance, const int shiftX, const int shiftY, const int windowDimension, const int grayLevel)
{
	GLCM output;
	initializeMetaGLCM(&output, distance, shiftX, shiftY, windowDimension, grayLevel);
	return output;
}

// Initialized MetaData of the GLCM
void initializeMetaGLCM(GLCM * glcm, const int distance, const int shiftX, const int shiftY, const int windowDimension, const int grayLevel)
{
	glcm->distance = distance;
	glcm->shiftX = shiftX;
	glcm->shiftY = shiftY;
	glcm->windowDimension = windowDimension;
	glcm->borderX = (windowDimension - (distance * shiftX));
	glcm->borderY = (windowDimension - (distance * shiftY));
	glcm->numberOfPairs = glcm->borderX * glcm->borderY;
	glcm->maxGrayLevel = grayLevel;

	// Inutile inizializzare questo campo
	glcm->numberOfUniquePairs = glcm->numberOfPairs;
}

// From a linearized vector of every pixel pair and an initialized GLCM
// it will generate the minimal representation of the MetaGLCM
void initializeMetaGLCMElements(struct GLCM * metaGLCM, const int * pixelPairs)
{
	// Allocate space for every different pair
	int * codifiedMatrix = (int *) malloc(sizeof(int) * metaGLCM->numberOfPairs);
	int k = 0;
	int referenceGrayLevel;
	int neighborGrayLevel;

	// Codify every single pair
	for (int i = 0; i < metaGLCM->borderY ; i++)
	{
		for (int j = 0; j < metaGLCM->borderX; j++)
		{
			// Extract the two pixels in the pair
			referenceGrayLevel = pixelPairs [(i * metaGLCM->windowDimension) + j];
			neighborGrayLevel = pixelPairs [(i + metaGLCM->shiftY) * metaGLCM->windowDimension + (j + metaGLCM->shiftX)];

			codifiedMatrix[k] = (((referenceGrayLevel * metaGLCM->maxGrayLevel) +
			neighborGrayLevel) * (metaGLCM->numberOfPairs)) ;
			k++;
		}
	}
	
	sort(codifiedMatrix, metaGLCM->numberOfPairs);
	int finalLength = localCompress(codifiedMatrix, metaGLCM->numberOfPairs);
	codifiedMatrix = (int *) realloc(codifiedMatrix, sizeof(int) * finalLength);
	metaGLCM->elements = codifiedMatrix;
	if(metaGLCM->elements == NULL){
		fprintf(stderr, "Couldn't Realloc the array while codifying\n");
		exit(-1);
	}
	metaGLCM->numberOfUniquePairs = finalLength;
}

/* 
	Compress codified gray pairs with different multiplicty into a single element
	Requires: sorted array with unique elements
*/
void compressMultiplicity(struct GLCM * metaGLCM)
{
	int countSimilar = 0, i=0 ,j=0; 
	GrayPair actual, next;
	while(i < metaGLCM->numberOfUniquePairs)
	{
		j=i+1;
		// WARNING L'UNPACK di 2 coppie diverse ha senso solo se ottenuti con lo stesso numberOfPairs
		actual = unPack(metaGLCM->elements[i], metaGLCM->numberOfPairs, metaGLCM->maxGrayLevel);
		next = unPack(metaGLCM->elements[j], metaGLCM->numberOfPairs, metaGLCM->maxGrayLevel);
		printPair(metaGLCM->elements[i], metaGLCM->numberOfPairs, metaGLCM->maxGrayLevel);
		while(compareEqualsGrayPairs(actual,next))
		{
			metaGLCM->elements[i] += next.multiplicity; 
			metaGLCM->elements[j] = INT_MAX; // Will be destroyed by sort&resize
			countSimilar++; 
			j++; // next element
			next = unPack(metaGLCM->elements[j], metaGLCM->numberOfPairs, metaGLCM->maxGrayLevel);
		}
		i=j;

	}
	sort(metaGLCM->elements, metaGLCM->numberOfUniquePairs);
	metaGLCM->numberOfUniquePairs = metaGLCM->numberOfUniquePairs - countSimilar;
}

/* 
	Will combine same aggregated pairs that differ for their multiplicity
*/
int compressAggregatedMultiplicity(int * summedPairs, int length, const int numberOfPairs)
{
	int countSimilar = 0, i=0 ,j=0; 
	AggregatedPair actual, next;

	while(i < length)
	{
		j=i+1;
		// WARNING L'UNPACK di 2 coppie diverse ha senso solo se ottenuti con lo stesso numberOfPairs
		actual = aggregatedPairUnPack(summedPairs[i], numberOfPairs);
		next = aggregatedPairUnPack(summedPairs[j], numberOfPairs);
		while(compareEqualsAggregatedPairs(actual, next))
		{
			summedPairs[i] += next.multiplicity; 
			summedPairs[j] = INT_MAX; // Will be destroyed by sort&resize
			countSimilar++; 
			j++; // next element
			next = aggregatedPairUnPack(summedPairs[j], numberOfPairs);
		}
		i=j;

	}
	int finalLength = length - countSimilar;
	sort(summedPairs, length); // Shift Exluded elements to right
	summedPairs = (int *) realloc(summedPairs, sizeof(int) * finalLength);
	return finalLength;
}

/*	Add to a metaGLCM an array of codified gray pairs
	After the insertion the same pair, codified with different 
	multiplicity, will be merged
*/
void addElements(struct GLCM * metaGLCM, int * elementsToAdd, int elementsLength)
{
	// New total number of elements
	int newSize = metaGLCM->numberOfPairs + elementsLength;

	// Physical space necessary to embedd actual list and new elements
	int enlargedLength = metaGLCM->numberOfUniquePairs + elementsLength;
	
	// Enlarge actual pair list
	metaGLCM->elements = (int *) realloc(metaGLCM->elements, sizeof(int) 
		* enlargedLength);
	printGLCMData(metaGLCM);

	// Copy the added elements in the new cells
	for(int i=0; i < elementsLength; i++)
	{
		metaGLCM->elements[i + metaGLCM->numberOfUniquePairs] =
			elementsToAdd[i];
	}
	sort(metaGLCM->elements, enlargedLength);

	/*
	// First, compress identical elements
	int reducedLength = localCompress(metaGLCM->elements, enlargedLength);
	metaGLCM->elements = (int *) realloc(metaGLCM->elements, sizeof(int) * reducedLength);
	metaGLCM->numberOfUniquePairs = reducedLength;
	printArray(metaGLCM->elements, reducedLength);
	printGLCMData(metaGLCM);
	// Second, compress same pair with different multiplicity
	compressMultiplicity(metaGLCM);
	metaGLCM->numberOfPairs = newSize; // Change after compression because it determines decodifcation of pairs
	printMetaGlcm(*metaGLCM);
	printGLCMData(metaGLCM);
	*/

}


// Reduce the molteplicity into the metaGLCM of elements given
// ASSUMPTION all given elements can be found in metaGLCM
void dwarf(int * metaGLCM, int * listElements, int lengthGlcm, int lengthElements)
{
	
}

/*
	From a metaGLCM will codify the sum of its grayPair levels
	outputList will contain aggregatedPairs (obtained with sum)
*/
int codifySummedPairs(const GLCM metaGLCM, int * outputList)
{
	GrayPair actualPair;
	// Navigate every unique gray pair and represent their sum 
	for (int i = 0; i < metaGLCM.numberOfUniquePairs; ++i)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, metaGLCM.maxGrayLevel);
		outputList[i] = (actualPair.grayLevelI + actualPair.grayLevelJ) * metaGLCM.numberOfPairs + actualPair.multiplicity -1;
	}
	// Need to compress identical generated elements
	sort(outputList, metaGLCM.numberOfUniquePairs);
	int finalLength = compressAggregatedMultiplicity(outputList, metaGLCM.numberOfUniquePairs, metaGLCM.numberOfPairs);

    std::cout << "\nSummed MetaGLCM: ";
    printAggrregatedMetaGlcm(outputList, finalLength, metaGLCM.numberOfPairs);

	return finalLength;
}

/*
	From a metaGLCM will codify the difference of its grayPair levels
	outputList will contain aggregatedPairs (obtained with difference)
*/
int codifySubtractedPairs(const GLCM metaGLCM, int * outputList)
{
	GrayPair actualPair;
	// Navigate every unique gray pair and represent their sum 
	for (int i = 0; i < metaGLCM.numberOfUniquePairs; ++i)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, metaGLCM.maxGrayLevel);
		outputList[i] = abs(actualPair.grayLevelI - actualPair.grayLevelJ) * metaGLCM.numberOfPairs + actualPair.multiplicity -1;
	}
	// Need to compress identical generated elements
	sort(outputList, metaGLCM.numberOfUniquePairs);
	int finalLength = compressAggregatedMultiplicity(outputList, metaGLCM.numberOfUniquePairs, metaGLCM.numberOfPairs);

	std::cout << "\nSubtracted MetaGLCM: ";
	printAggrregatedMetaGlcm(outputList, finalLength, metaGLCM.numberOfPairs);

	return finalLength;
}