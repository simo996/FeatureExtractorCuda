#include <stdio.h>
#include "AggregatedGrayPair.h"

__device__  AggregatedGrayPair::AggregatedGrayPair() {
    grayLevel = 0;
    frequency = 0;
}

__device__ AggregatedGrayPair::AggregatedGrayPair(unsigned int i, unsigned int freq){
    grayLevel = i;
    frequency = freq;
}

__device__ void AggregatedGrayPair::printPair() const {
    printf("k: %d", grayLevel);
    printf("\tmult: %d", frequency);
    printf("\n");
}

__device__ bool AggregatedGrayPair::compareTo(AggregatedGrayPair other) const{
    return (grayLevel == other.getAggregatedGrayLevel());
}

/* Extracting pairs */
__device__ int AggregatedGrayPair::getAggregatedGrayLevel() const{
    return grayLevel;
}

__device__ unsigned int AggregatedGrayPair::getFrequency() const {
    return frequency;
}

__device__ void AggregatedGrayPair::increaseFrequency(unsigned int amount){
    frequency += amount;
}
