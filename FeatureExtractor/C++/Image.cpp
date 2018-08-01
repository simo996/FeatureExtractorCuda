#include <iostream>
#include "Image.h"

int Image::getRows() const{
    return rows;
}

int Image::getColumns() const{
    return columns;
}

const int* Image::getPixels() const{
    return pixels;
}

int Image::getMaxGrayLevel() const{
    return maxGrayLevel;
}

void Image::printElements() const{
    std::cout << "Img = " << std::endl;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            std::cout << pixels[i * rows + j] << " " ;
        }
        std::cout << std::endl;
    }

}