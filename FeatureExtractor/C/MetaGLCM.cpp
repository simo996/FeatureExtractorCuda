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
struct GLCM initializeMetaGLCM(const int distance, const int shiftX, const int shiftY, const int windowRows, const int windowColumns)
{
    GLCM output;
    output.distance = distance;
    output.shiftX = shiftX;
    output.shiftY = shiftY;
    output.borderX = (windowColumns - (distance * shiftX));
    output.borderY = (windowRows - (distance * shiftY));
    output.numberOfPairs = output.borderX * output.borderY;

    // Inutile inizializzare questo campo
    output.numberOfUniquePairs = output.numberOfPairs;
    return output;
}

// From a linearized vector of every pixel pair and an initialized GLCM
// it will generate the minimal representation of the MetaGLCM
void initializeMetaGLCMElements(struct GLCM metaGLCM, int * pixelPairs, int grayLevel)
{
    // Allocate space for every different pair
    int * codifiedMatrix = (int *) malloc(sizeof(int) * metaGLCM.numberOfPairs);
    int k = 0;
    int referenceGrayLevel;
    int neighborGrayLevel;

    // Codify every single pair
    for (int i = 0; i < metaGLCM.borderY ; i++)
    {
        for (int j = 0; j < metaGLCM.borderX; j++)
        {

            referenceGrayLevel = getElementFromLinearMatrix(pixelPairs, 
                metaGLCM.borderY, metaGLCM.borderX, i, j);
            neighborGrayLevel = getElementFromLinearMatrix(pixelPairs, 
                metaGLCM.borderY, metaGLCM.borderX, i + metaGLCM.shiftY, j + metaGLCM.shiftX);

            codifiedMatrix[k] = (((referenceGrayLevel * grayLevel) +
            neighborGrayLevel) * (metaGLCM.numberOfPairs)) ;
            k++;
        }
    }

    metaGLCM.numberOfUniquePairs = localCompress(codifiedMatrix, k);
    // IS ALLOCATION INSIDE A FUNCTION SAFE ???
    metaGLCM.elements = (int *) malloc(sizeof(int) * metaGLCM.numberOfUniquePairs);
    memcpy(metaGLCM.elements, codifiedMatrix, metaGLCM.numberOfUniquePairs * sizeof(int));
    free(codifiedMatrix);
}

// Same pair with different multeplicty collapsed into a single element
int compressMultiplicity(int * inputArray, const int length, const int numberOfPairs, const int imgGrayLevel){
    int occurrences = 0;
    int deletions = 0;
    int j = 1;

    for (int i = 0; i < length; i++)
    {
        occurrences = 0;
        j = i+1;
        // Count multiple occurrences of the same number
        //DIVISIONE PER ZERO, PER GIOVE, CRASHA
        while((inputArray[i] != INT_MAX) && (inputArray[i] % inputArray [j]) < numberOfPairs)
        {
            occurrences+=unPack(inputArray[j],numberOfPairs,imgGrayLevel).multiplicity;
            deletions++;
            inputArray[j] = INT_MAX; // for destroying
            j++;
        }
        // Increment quantity
        if(occurrences>=numberOfPairs){
            fprintf(stderr, "Error while compressing minding the multiplicity\n");
            fprintf(stderr, "Summed Multiplicity: %d\n",occurrences);
            fprintf(stderr, "Element: %d\n",inputArray[i]);

            exit(-1);
        }
        if(inputArray[j] != INT_MAX){
            inputArray[i] = inputArray[i]+occurrences;
        }

    }

    sort(inputArray,length);
    // After |length| numbers there should be only INT_MAX
    return length-deletions;

}

// Adapt the GLCM to include ulterior elements already codified
// Will return the modified array and its length
int addElements(int * metaGLCM, int * elementsToAdd, int * outputArray, const int initialLength, const int numElements, const int numberOfPairs, const int grayLevel)
{

    // Combine the 2 arrays
    int sumLength= numElements + initialLength;

    int i,j;
    for ( i = 0; i < initialLength; ++i) {
        outputArray[i] = metaGLCM[i];
    }
    for(j=0; j< numElements; j++)
    {
        outputArray[i]= elementsToAdd[j];
        i++;
    }
    printArray(outputArray,sumLength);

    // Sort and compress identical pairs
    sort(outputArray, sumLength); // Required for successive compresses
    int reducedLength = localCompress(outputArray, sumLength);
    printArray(outputArray,reducedLength);


    // Same pair with different molteplicity will be compressed

    // PERCHÈ NON RITORNA???
    int finalReducedLength = compressMultiplicity(outputArray,reducedLength,numberOfPairs,grayLevel);
    printArray(outputArray,finalReducedLength);
    return finalReducedLength;
}


// Reduce the molteplicity into the metaGLCM of elements given
// ASSUMPTION all given elements can be found in metaGLCM
void dwarf(int * metaGLCM, int * listElements, int lengthGlcm, int lengthElements)
{
    // both arrays need to be ordered
    sort(listElements, lengthElements);
    int j = 0;

    for (int i = 0; i < lengthElements; ++i)
    {
        while(metaGLCM[j] != listElements [i]){
            j++;
        }
        //int multiplicity = listElements[i].unPack(...).multiplicity;
        // metaGLCM[j]-=multiplicity;
    }
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
        outputList[i] = (actualPair.grayLevelI+actualPair.grayLevelJ) * metaGLCM.numberOfPairs;
    }
    // Need to compress identical generated elements
    sort(outputList, metaGLCM.numberOfPairs);
    int finalLength = localCompress(outputList, metaGLCM.numberOfPairs);
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
        outputList[i] = abs(actualPair.grayLevelI - actualPair.grayLevelJ) * metaGLCM.numberOfPairs;
    }
    // Need to compress identical generated elements
    sort(outputList, metaGLCM.numberOfPairs);
    int finalLength = localCompress(outputList, metaGLCM.numberOfPairs);
    return finalLength;
}
