//
// Created by simone on 12/08/18.
//


#include "ImageAllocated.h"

unsigned int ImageAllocated::getRows() const{
    return rows;
}

unsigned int ImageAllocated::getColumns() const{
    return columns;
}

const vector<unsigned int> ImageAllocated::getPixels() const{
    return pixels;
}

unsigned int ImageAllocated::getMaxGrayLevel() const{
    return maxGrayLevel;
}

void ImageAllocated::printElements() const {
    std::cout << "Img = " << std::endl;

    for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < columns; j++) {
            std::cout << pixels[i * rows + j] << " ";
        }
        std::cout << std::endl;
    }
}