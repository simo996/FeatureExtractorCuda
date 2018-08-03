#include <iostream>
#include "Image.h"

uint Image::getRows() const{
    return rows;
}

uint Image::getColumns() const{
    return columns;
}

const uint * Image::getPixels() const{
    return pixels;
}

uint Image::getMaxGrayLevel() const{
    return maxGrayLevel;
}

void Image::printElements() const{
    std::cout << "Img = " << std::endl;

    for (uint i = 0; i < rows; i++)
    {
        for (uint j = 0; j < columns; j++)
        {
            std::cout << pixels[i * rows + j] << " " ;
        }
        std::cout << std::endl;
    }

}