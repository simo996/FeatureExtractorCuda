/*
 * ImageData.cpp
 *
 *  Created on: 26/ago/2018
 *      Author: simone
 */

#include "ImageData.h"


__host__ __device__ unsigned int ImageData::getRows() const{
    return rows;
}

__host__ __device__ unsigned int ImageData::getColumns() const{
    return columns;
}

__host__ __device__ unsigned int ImageData::getMaxGrayLevel() const{
    return maxGrayLevel;
}

__host__ __device__ void ImageData::printElements(unsigned int* pixels) const {
    printf("Img = \n");

    for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < columns; j++) {
            printf("%d ", pixels[i * rows + j]);
        }
        printf("\n");
    }
}
