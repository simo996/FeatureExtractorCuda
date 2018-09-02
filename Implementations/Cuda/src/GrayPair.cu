/*
 * GrayPair.cpp
 *
 *  Created on: 25/ago/2018
 *      Author: simone
 */
#include <stdio.h>
#include "GrayPair.h"

/* Constructors*/
__device__ GrayPair::GrayPair()
{
    grayLevelI = 0;
    grayLevelJ = 0;
    frequency = 0;
}

__device__ GrayPair::GrayPair (grayLevelType i, grayLevelType j) {
   grayLevelI = i;
   grayLevelJ = j;
   frequency = 1;
}

__device__ void GrayPair::frequencyIncrease(){
    frequency+=1;
}

__device__ bool GrayPair::compareTo(GrayPair other) const{
    if((grayLevelI == other.getGrayLevelI())
    && (grayLevelJ == other.getGrayLevelJ()))
        return true;
    else
        return false;
}

__device__ void GrayPair::printPair()const {
    printf("i: %d", grayLevelI);
    printf("\tj: %d", grayLevelJ);
    printf("\tmult: %d", frequency);
    printf("\n");
}

/* Extracting pairs */
__device__ grayLevelType GrayPair::getGrayLevelI() const{
    return grayLevelI;
}

__device__ grayLevelType GrayPair::getGrayLevelJ() const{
    return grayLevelJ;
}

__device__ frequencyType GrayPair::getFrequency() const {
    return frequency;
}
