//
// Created by simo on 11/07/18.
//

#include <iostream>
#include "GrayPair.h"

/* Constructors*/
GrayPair::GrayPair (int i, int j) {
   grayLevelI = i;
   grayLevelJ = j;
}

void GrayPair::printPair()const {
    std::cout << "i: "<< grayLevelI;
    std::cout << "\tj: " << grayLevelJ;
}

/* Extracting pairs */
int GrayPair::getGrayLevelI() const{
    return grayLevelI;
}

int GrayPair::getGrayLevelJ() const{
    return grayLevelJ;
}
