#include <iostream>
#include <stdio.h>
#include <cstring>
#include "ImageData.h"



uint ImageData::getRows() const{
    return rows;
}

uint ImageData::getColumns() const{
    return columns;
}

uint ImageData::getMaxGrayLevel() const{
    return maxGrayLevel;
}

void ImageData::printElements(unsigned int* pixels) const {
    std::cout << "Img = " << std::endl;

    for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < columns; j++) {
            std::cout << pixels[i * rows + j] << " ";
        }
        std::cout << std::endl;
    }
}