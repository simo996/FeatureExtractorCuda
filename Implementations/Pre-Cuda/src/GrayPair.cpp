//
// Created by simo on 11/07/18.
//

#include <iostream>
#include "GrayPair.h"

/* Constructors*/
GrayPair::GrayPair()
{
    grayLevelI = 0;
    grayLevelJ = 0;
    frequency = 0;
}

GrayPair::GrayPair (grayLevelType i, grayLevelType j) {
   grayLevelI = i;
   grayLevelJ = j;
   frequency = 1;
}

void GrayPair::printPair()const {
    std::cout << "i: "<< grayLevelI;
    std::cout << "\tj: " << grayLevelJ;
    std::cout << "\tmult: " << frequency;
    std::cout << std::endl;
}

void GrayPair::frequencyIncrease(){
    frequency += 1;
}

bool GrayPair::compareTo(GrayPair other) const{
    if((grayLevelI == other.getGrayLevelI())
    && (grayLevelJ == other.getGrayLevelJ()))
        return true;
    else
        return false;
}

/* Extracting pairs */
grayLevelType GrayPair::getGrayLevelI() const{
    return grayLevelI;
}

grayLevelType GrayPair::getGrayLevelJ() const{
    return grayLevelJ;
}

frequencyType GrayPair::getFrequency() const {
    return frequency;
}
