//
// Created by simo on 11/07/18.
//

#include <iostream>
#include "GrayPair.h"

/* Constructors*/
GrayPair::GrayPair (unsigned int i, unsigned int j) {
   grayLevelI = i;
   grayLevelJ = j;
}

void GrayPair::printPair()const {
    std::cout << "i: "<< grayLevelI;
    std::cout << "\tj: " << grayLevelJ;
}

/* Extracting pairs */
uint GrayPair::getGrayLevelI() const{
    return grayLevelI;
}

uint GrayPair::getGrayLevelJ() const{
    return grayLevelJ;
}
