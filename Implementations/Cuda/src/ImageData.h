/*
 * ImageData.h
 *
 *  Created on: 26/ago/2018
 *      Author: simone
 */

#ifndef IMAGEDATA_H_
#define IMAGEDATA_H_

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__ 
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif

#include "Image.h"

using namespace std;

class ImageData {
public:
    CUDA_HOSTDEV ImageData(unsigned int rows, unsigned int columns, unsigned int mxGrayLevel)
            : rows(rows), columns(columns), maxGrayLevel(mxGrayLevel){};
    CUDA_HOSTDEV ImageData(Image& img)
                : rows(img.getRows()), columns(img.getColumns()), maxGrayLevel(img.getMaxGrayLevel()){};
    CUDA_HOSTDEV unsigned int getRows() const;
    CUDA_HOSTDEV unsigned int getColumns() const;
    CUDA_HOSTDEV unsigned int getMaxGrayLevel() const;
    CUDA_HOSTDEV void printElements(unsigned int* pixels) const;

private:
    const unsigned int rows;
    const unsigned int columns;
    const unsigned int maxGrayLevel;
};


#endif /* IMAGEDATA_H_ */
