//
// Created by simo on 11/07/18.
//

#include <iostream>
#include "AggregatedGrayPair.h"

AggregatedGrayPair::AggregatedGrayPair() {
    grayLevel = 0;
    frequency = 0;
}

AggregatedGrayPair::AggregatedGrayPair(unsigned int i, unsigned int freq){
    grayLevel = i;
    frequency = freq;
}

void AggregatedGrayPair::printPair() const {
    std::cout << "k: " << grayLevel;
    std::cout << "\tmult: " << frequency;
    std::cout << std::endl;
}

/* Extracting pairs */
int AggregatedGrayPair::getAggregatedGrayLevel() const{
    return grayLevel;
}

unsigned int AggregatedGrayPair::getFrequency() const {
    return frequency;
}

bool AggregatedGrayPair::compareTo(AggregatedGrayPair other) const{
    return (grayLevel == other.getAggregatedGrayLevel());
}

void AggregatedGrayPair::increaseFrequency(unsigned int amount){
    frequency += amount;
}


