//
// Created by simo on 07/05/18.
//

#include <iostream>
#include <climits>
#include <stdlib.h>
#include <cstdio>
#include <cstring>
#include <assert.h>

#include "MetaGLCM.h"
#include "SupportCode.h"
#include "GrayPair.h"


void printGLCMData(const GLCM input)
{
	std::cout << std::endl;
	std::cout << "Shift rows : " << input.shiftRows << std::endl;
	std::cout << "Shift columns: " << input.shiftColumns  << std::endl;
	std::cout << "Father Window dimension: "<< input.windowDimension  << std::endl;
	std::cout << "Border Rows: "<< input.borderRows  << std::endl;
	std::cout << "Border Columns: " << input.borderColumns  << std::endl;
	std::cout << "Simmetric: ";
	if(input.simmetric){
		std::cout << "Yes" << std::endl;
	}
	else{
		std::cout << "No" << std::endl;
	}

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

/*	INITIALIZATION METHODS 
	Obtain essential parameters for constructing MetaGLMC
	Optional parameter: simmetric (default = NO)

*/
// Implicit struct version
struct GLCM initializeMetaGLCM(const int distance, const int shiftX, const int shiftY, const int windowDimension, const int grayLevel)
{
	GLCM output;
	initializeMetaGLCM(&output, distance, shiftX, shiftY, windowDimension, grayLevel, false);
<<<<<<< HEAD
	return output;
}

// Explicit struct version
struct GLCM initializeMetaGLCM(const int distance, const int shiftX, const int shiftY, const int windowDimension, const int grayLevel, bool simmetric)
{
	GLCM output;
	initializeMetaGLCM(&output, distance, shiftX, shiftY, windowDimension, grayLevel, simmetric);
	return output;
}

=======
	return output;
}

// Explicit struct version
struct GLCM initializeMetaGLCM(const int distance, const int shiftX, const int shiftY, const int windowDimension, const int grayLevel, bool simmetric)
{
	GLCM output;
	initializeMetaGLCM(&output, distance, shiftX, shiftY, windowDimension, grayLevel, simmetric);
	return output;
}

>>>>>>> e1d9826bc5e8fe12062720d0cd74c0580105ff13
// Implicit pointer version
void initializeMetaGLCM(GLCM * glcm, const int distance, const int shiftX, const int shiftY, const int windowDimension, const int grayLevel)
{
	initializeMetaGLCM(glcm, distance, shiftX, shiftY, windowDimension, grayLevel,
		false); // DEFAULT = ASIMMETRIC
}

// Explicit pointer version
void initializeMetaGLCM(GLCM * glcm, const int distance, const int shiftX, const int shiftY, const int windowDimension, const int grayLevel, bool simmetric)
{
	glcm->distance = distance;
	glcm->shiftRows = shiftX;
	glcm->shiftColumns = shiftY;
	glcm->windowDimension = windowDimension;
	glcm->borderColumns = (windowDimension - (distance * abs(shiftY)));
	glcm->borderRows = (windowDimension - (distance * abs(shiftX)));
	glcm->simmetric = simmetric;
	glcm->numberOfPairs = glcm->borderRows * glcm->borderColumns;
	glcm->maxGrayLevel = grayLevel;

	// Inutile inizializzare questo campo
	glcm->numberOfUniquePairs = glcm->numberOfPairs;
}

// From a linearized vector of every pixel pair and an initialized GLCM
// it will generate the minimal representation of the MetaGLCM


void initializeMetaGLCMElements(struct GLCM * metaGLCM, const int * pixelPairs)
{
	// Allocate space for every different pair
	int actualNumberOfPairs;

	if(metaGLCM->simmetric)
	{	
		// 2 Gray pairs from each pixel pair <i,j> and <j,i> 
		actualNumberOfPairs = 2 * metaGLCM->numberOfPairs;
	}
	else
	{
		actualNumberOfPairs = metaGLCM->numberOfPairs;
	}
	int * codifiedMatrix = (int *) malloc(sizeof(int) * actualNumberOfPairs);
	
	int k = 0;
	int referenceGrayLevel;
	int neighborGrayLevel;

	// Define subBorders offset depending on orientation
	int initialColumnOffset = 0; // for 0°,45°,90°
	if((metaGLCM->shiftRows * metaGLCM->shiftColumns) > 0) // 135°
		initialColumnOffset = 1;
	int initialRowOffset = 1; // for 45°,90°,135°
	if((metaGLCM->shiftRows == 0) && (metaGLCM->shiftColumns > 0))
		initialRowOffset = 0; // for 0°
	//std::cout << "DEBUG -\tRowOffset: " << initialRowOffset << "\tColOffset: " << initialColumnOffset;
	// Codify every single pair
	for (int i = 0; i < metaGLCM->borderRows ; i++)
	{
		for (int j = 0; j < metaGLCM->borderColumns; j++)
		{
			// Extract the two pixels in the pair
			referenceGrayLevel = pixelPairs [((i + initialRowOffset) * metaGLCM->windowDimension) + (j + initialColumnOffset)];
			neighborGrayLevel = pixelPairs [((i + initialRowOffset) + metaGLCM->shiftRows) * metaGLCM->windowDimension + (j + initialColumnOffset + metaGLCM->shiftColumns)];

			codifiedMatrix[k] = (((referenceGrayLevel * metaGLCM->maxGrayLevel) + 
				neighborGrayLevel) * (metaGLCM->numberOfPairs)) ;
			if(metaGLCM->simmetric)
			{
				codifiedMatrix[k+1] = (((neighborGrayLevel * metaGLCM->maxGrayLevel) + 
					referenceGrayLevel) * (metaGLCM->numberOfPairs)) ;
				k++;
			}
			k++;
		}
	}
	sort(codifiedMatrix, actualNumberOfPairs);
	int finalLength = localCompress(codifiedMatrix, actualNumberOfPairs);
	assert(finalLength > 0);
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
int compressAggregatedMultiplicity(int * aggregatedPairs, int length, const int numberOfPairs)
{
	int countSimilar = 0, i=0 ,j=0; 
	AggregatedPair actual, next;

	while(i < length)
	{
		j=i+1;
		// WARNING L'UNPACK di 2 coppie diverse ha senso solo se ottenuti con lo stesso numberOfPairs
		actual = aggregatedPairUnPack(aggregatedPairs[i], numberOfPairs);
		next = aggregatedPairUnPack(aggregatedPairs[j], numberOfPairs);
		while(compareEqualsAggregatedPairs(actual, next))
		{
			aggregatedPairs[i] += next.multiplicity; 
			aggregatedPairs[j] = INT_MAX; // Will be destroyed by sort&resize
			countSimilar++; 
			j++; // next element
			next = aggregatedPairUnPack(aggregatedPairs[j], numberOfPairs);
		}
		i=j;

	}
	int finalLength = length - countSimilar;
	assert(finalLength > 0);
	sort(aggregatedPairs, length); // Shift Exluded elements to right
    int * newDataPointer = (int *) realloc(aggregatedPairs, sizeof(int) * finalLength);
    if (!newDataPointer){
        printf("\n\nGROSSO BRUTTO ERRORE D'ALLOCAZIONE");
    }
    else
    {
        aggregatedPairs = newDataPointer;
    }
	return finalLength;
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
    assert(finalLength > 0);
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
    assert(finalLength > 0);

	std::cout << "\nSubtracted MetaGLCM: ";
	printAggrregatedMetaGlcm(outputList, finalLength, metaGLCM.numberOfPairs);

	return finalLength;
}

