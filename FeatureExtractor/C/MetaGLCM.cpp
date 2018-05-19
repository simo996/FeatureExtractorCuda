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

void printMetaGlcm(const int * metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel)
{
	std::cout << std::endl;
	for (int i = 0; i < length; ++i)
	{
		printPair(metaGLCM[i], numberOfPairs, maxGrayLevel);
	}
	std::cout << std::endl;

}

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

// Object version
void printMetaGlcm(const struct GLCM metaGLCM, const int maxGrayLevel)
{
	std::cout << std::endl;
	for (int i = 0; i < metaGLCM.numberOfUniquePairs; ++i)
	{
		printPair(metaGLCM.elements[i], metaGLCM.numberOfPairs, maxGrayLevel);
	}
	std::cout << std::endl;

}

// Initialized MetaData of the GLCM
struct GLCM initializeMetaGLCM(const int distance, const int shiftX, const int shiftY, const int windowDimension)
{
	GLCM output;
	initializeMetaGLCM(&output, distance, shiftX, shiftY, windowDimension);
	return output;
}

// Initialized MetaData of the GLCM
void initializeMetaGLCM(GLCM * glcm, const int distance, const int shiftX, const int shiftY, const int windowDimension)
{
	glcm->distance = distance;
	glcm->shiftX = shiftX;
	glcm->shiftY = shiftY;
	glcm->windowDimension = windowDimension;
	glcm->borderX = (windowDimension - (distance * shiftX));
	glcm->borderY = (windowDimension - (distance * shiftY));
	glcm->numberOfPairs = glcm->borderX * glcm->borderY;

	// Inutile inizializzare questo campo
	glcm->numberOfUniquePairs = glcm->numberOfPairs;
}

// From a linearized vector of every pixel pair and an initialized GLCM
// it will generate the minimal representation of the MetaGLCM
void initializeMetaGLCMElements(struct GLCM * metaGLCM, const int * pixelPairs, const int grayLevel)
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
			// TODO FIX INCORRECT ALGORITHM
					/* incongruence between window's fixed size for getting an element
					 * and sub-borders depending on the dimension used
					*/
			// Extract the two pixels in the pair
			referenceGrayLevel = pixelPairs [(i * metaGLCM->windowDimension) + j];
			neighborGrayLevel = pixelPairs [(i + metaGLCM->shiftY) * metaGLCM->windowDimension + (j + metaGLCM->shiftX)];

			codifiedMatrix[k] = (((referenceGrayLevel * grayLevel) +
			neighborGrayLevel) * (metaGLCM->numberOfPairs)) ;
			k++;
		}
	}
	
	sort(codifiedMatrix, metaGLCM->numberOfPairs);
	int finalLength = localCompress(codifiedMatrix, metaGLCM->numberOfPairs);
	codifiedMatrix = (int *) realloc(codifiedMatrix, sizeof(int) * finalLength);
	metaGLCM->elements = codifiedMatrix;
	metaGLCM->numberOfUniquePairs = finalLength;
}

/* 
	Compress codified gray pairs with different multeplicty into a single element
	Requires: sorted array with unique elements
*/
void compressMultiplicity(struct GLCM * metaGLCM, const int imgGrayLevel){

}

/* 
	Add to a metaGLCM an array of codified gray pairs
	After the insertion the same pair, codified with different 
	multiplicity, will be merged
*/
void addElements(struct GLCM * metaGLCM, int * elementsToAdd, int elementsLength, const int imgGrayLevel)
{
	int incrementedLength = metaGLCM->numberOfUniquePairs + elementsLength;

	metaGLCM->elements = (int *) realloc(metaGLCM->elements, sizeof(int) * incrementedLength);
	if(metaGLCM->elements == NULL){
		fprintf(stderr, "Couldn't Realloc the array\n");
		exit(-1);
	}
	// Copy the added elements in the new cells
	for(int i=0; i < elementsLength; i++)
	{
		metaGLCM->elements[i + metaGLCM->numberOfUniquePairs] =
			elementsToAdd[i];
	}
	printArray(metaGLCM->elements, incrementedLength);
	sort(metaGLCM->elements, incrementedLength);
	printArray(metaGLCM->elements, incrementedLength);
	// First, compress identical elements
	int reducedLength = localCompress(metaGLCM->elements, incrementedLength);
	metaGLCM->elements = (int *) realloc(metaGLCM->elements, sizeof(int) * reducedLength);
	metaGLCM->numberOfPairs = reducedLength;
	printArray(metaGLCM->elements, reducedLength);
	// Second, compress same pair with different multiplicity

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
int codifySummedPairs(const GLCM metaGLCM, int * outputList, const int maxGrayLevel )
{
	GrayPair actualPair;
	// Navigate every unique gray pair and represent their sum 
	for (int i = 0; i < metaGLCM.numberOfUniquePairs; ++i)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, maxGrayLevel);
		outputList[i] = (actualPair.grayLevelI+actualPair.grayLevelJ) * metaGLCM.numberOfPairs + actualPair.multiplicity;
	}
	// Need to compress same pairs, even with different multiplicity
	sort(outputList, metaGLCM.numberOfUniquePairs);
	int finalLength = localCompress(outputList, metaGLCM.numberOfUniquePairs);
	return finalLength;
}

/*
	From a metaGLCM will codify the difference of its grayPair levels
	outputList will contain aggregatedPairs (obtained with difference)
*/
int codifySubtractedPairs(const GLCM metaGLCM, int * outputList, const int maxGrayLevel )
{
	GrayPair actualPair;
	// Navigate every unique gray pair and represent their sum 
	for (int i = 0; i < metaGLCM.numberOfUniquePairs; ++i)
	{
		actualPair = unPack(metaGLCM.elements[i], metaGLCM.numberOfPairs, maxGrayLevel);
		outputList[i] = abs(actualPair.grayLevelI - actualPair.grayLevelJ) * metaGLCM.numberOfPairs + actualPair.multiplicity;
	}
	printArray(outputList, metaGLCM.numberOfUniquePairs);
	// Need to compress identical generated elements
	sort(outputList, metaGLCM.numberOfUniquePairs);
	printArray(outputList, metaGLCM.numberOfUniquePairs);
	int finalLength = localCompress(outputList, metaGLCM.numberOfUniquePairs);
	printArray(outputList, finalLength);

	return finalLength;
}