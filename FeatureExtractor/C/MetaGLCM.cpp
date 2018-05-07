//
// Created by simo on 07/05/18.
//

#include <iostream>
#include <climits>
#include <stdlib.h>
#include <cstdio>

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



// Add identical elements into a new element
// Return the length of the compressed metaglcm
int compress(int * inputArray, int * outputArray, const int length)
{
    int occurrences = 0;
    int deletions = 0;
    int j = 1;

    for (int i = 0; i < length; i++)
    {
        occurrences = 0;
        j = i+1;
        // Count multiple occurrences of the same number
        while((inputArray[i] != -1) && (inputArray[i] == inputArray [j]))
        {
            occurrences++;
            deletions++;
            inputArray[j] = -1; // destroy from collection
            j++;
        }
        // Increment quantity
        inputArray[i]=inputArray[i]+occurrences;

    }

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
    return j;
}

// Same as compress but changing the InputArray and returning the final length
int localCompress(int * inputArray, const int length)
{
    int occurrences = 0;
    int deletions = 0;
    int j = 1;

    for (int i = 0; i < length; i++)
    {
        occurrences = 0;
        j = i+1;
        // Count multiple occurrences of the same number
        while((inputArray[i] != INT_MAX) && (inputArray[i] == inputArray [j]))
        {
            occurrences++;
            deletions++;
            inputArray[j] = INT_MAX; // for destroying
            j++;
        }
        // Increment quantity
        if(inputArray[j] != INT_MAX){
            inputArray[i]=inputArray[i]+occurrences;
        }

    }

    sort(inputArray,length);
    // After |length| numbers there should be only INT_MAX
    return length-deletions;
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

int codifySummedPairs(const int * metaGLCM, int * outputList, const int elements, const int numberOfPairs, const int maxGrayLevel ){
    int finalLength;

    // TODO

    return finalLength;
}

int codifySubtractedPairs(const int * metaGLCM, int * outputList, const int elements, const int numberOfPairs, const int maxGrayLevel ){
    int finalLength;

    // TODO


    return finalLength;
}
