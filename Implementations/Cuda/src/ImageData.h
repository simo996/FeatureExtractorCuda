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

/*
 * This class embeds metadata about the acquired image:
 * - pysical dimensions (height, width as rows and columns)
 * - the maximum gray level that could be encountered according to its type
*/

class ImageData {
public:
    CUDA_HOSTDEV ImageData(unsigned int rows, unsigned int columns, unsigned int mxGrayLevel)
            : rows(rows), columns(columns), maxGrayLevel(mxGrayLevel){};
    // Strip metadata from the "complete" Image class
    CUDA_HOSTDEV ImageData(Image& img)
                : rows(img.getRows()), columns(img.getColumns()), maxGrayLevel(img.getMaxGrayLevel()){};
    // Getters
    CUDA_HOSTDEV unsigned int getRows() const;
    CUDA_HOSTDEV unsigned int getColumns() const;
    CUDA_HOSTDEV unsigned int getMaxGrayLevel() const;   
    // Debug method
    CUDA_HOSTDEV void printElements(unsigned int* pixels) const;

private:
    const unsigned int rows;
    const unsigned int columns;
    const unsigned int maxGrayLevel;
};


#endif /* IMAGEDATA_H_ */
