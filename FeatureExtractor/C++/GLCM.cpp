#include <stdio.h>
#include <stdlib.h>
// OpenCv Libraries for loading MRImages
#include "GLCM.h"

// shiftX, shiftY given from the creator (the image Window)
void initializeData(int shiftX, int shiftY, int grayLevels, int windowColumns, int windowRows);
{
	distance = 1; // Always 1 
	this.shiftX = shiftX;
	this.shiftY = shiftY;
	this.maxGrayLevel = grayLevels; // image depth 
	computeBorders();
	numberOfPairs = computePairsNumber();
	borderX = computeBorderX(int windowColumns, int windowRows);
	borderY = computeBorderY(int windowColumns, int windowRows);
}

int computeBorderX(int windowColumns, int windowRows)
{
	int border = windowColumns - (distance * shiftX);
	return border;
}

int computeBorderY(int windowColumns, int windowRows)
{
	int border = window.Rows - (glcm0.distance * glcm0.shiftY);
	return border;
}


int computePairsNumber()
{
	numberOfPairs = borderX * borderY;
	return numberOfPairs;
}

// Actual work

// Create initial representation codifying each physical pair present in the window
int codify(int * imageElements, int length){
	// IMAGEELEMENTS MUST BE A LINARIZED MATRIX of the pixels
	int * codifiedMatrix=(int *) malloc(sizeof(int)*numberOfPairs);
	int k=0;
	int referenceGrayLevel;
	int neighborGrayLevel;

	// FIRST STEP: codify all pairs
	for (int i = 0; i < borderY ; i++) // rows
	{
		for (int j = 0; j < borderX; j++) // columns
		{
			referenceGrayLevel = imageElements[i][j];
			neighborGrayLevel = imageElements[i+shiftY][j+shiftX];

			codifiedMatrix[k] = ((referenceGrayLevel*maxGrayLevel) + 
			neighborGrayLevel) * (numberOfPairs+1); // +1 teoricamente non serve
			k++;
		}
	}

	sort(codifiedMatrix, numberOfPairs);
	// First create a big support array for copying non -1 elements in first positions
	int * compressedGLCM = (int *) malloc (sizeof(int)*numberOfPairs); // some dimension in excess
	int finalMetaGlcmLength = compress(codifiedMatrix, compressedGLCM, numberOfPairs);
	// then cut to useful length
	metaGLCM[finalMetaGlcmLength];
	memcpy(metaGLCM, compressedGLCM, finalMetaGlcmLength * sizeof(int));
	free(codifiedMatrix);
	free(compressedGLCM);
}

void sort(){
	sort(metaGLCM, length);
}

// Will reduce MetaGLCM to only useful elements
void compress(int * inputArray, int * outputArray, const int length)
{
	int occurrences = 0;
	int deletions = 0;
	int j = 1;

	// Pass all elements 
	for (int i = 0; i < length; i++)
	{
		occurrences = 0;
		j = i+1;
		// Count multiple occurrences of the same number
		while((inputArray[i] != -1) && (inputArray[i] == inputArray [j]))
		{
			occurrences++;
			deletions++;
			inputArray[j] = -1; // signal to destroy from collection
			j++;
		}
		// Increment quantity
		inputArray[i] = inputArray[i] + occurrences;

	}
	// TODO memory optimization
	// Move locally non -1 elements in the first positions and return useful length
	// locally = in inputArray -> no need of outputarray
	j = 0; // in the end equals to Length-deletions
	// Copy non -1 numbers in the output vector
	for (int i = 0; i < length; i++)
	{
		if(inputArray[i] != -1)
		{
			outputArray[j] = inputArray[i];
			j++;
		}
	}
	length = j;
}



// Reduce the molteplicity into the metaGLCM of elements given
// ASSUMPTION all given elements can be found in metaGLCM
// Return the length of the reduced and compressed list
void dwarf(int * listElements, int lengthElements)
{
	// both arrays need to be ordered
	sort(listElements, lengthElements);
	int j = 0;
	bool needToCompress = false;
	for (int i = 0; i < lengthElements; ++i)
	{
		while(metaGLCM[j] != listElements [i]){
			j++;
		}
		if(metaGLCM[j] == listElements [i]){
			GrayPair actual, element;
			// element = listElements[i].unPack(...);
			// actual = metaGLCM[j].unPack();
		
			if(actual.multeplicity < element.multeplicity){
				fprintf(stderr, "This should never happen; reduced too much of the multeplicity\n");
				exit(-1);
			}
			if(actual.multeplicity == element.multeplicity){
			{
				metaGLCM[j] =- 1;
				needToCompress = true;
			}
		}
		else
		{
			fprintf(stderr, "This should never happen; mismatch while dwarfing\n");
			exit(-1);
		}
	}
	if (needToCompress)
	{
		compress(...);
	}
}
