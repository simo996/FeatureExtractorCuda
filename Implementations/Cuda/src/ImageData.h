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
    CUDA_HOSTDEV explicit ImageData(unsigned int rows, unsigned int columns, int borders,
            unsigned int mxGrayLevel)
            : rows(rows), columns(columns), appliedBorders(borders),
            maxGrayLevel(mxGrayLevel){};
    // Strip metadata from the "complete" Image class
    CUDA_HOSTDEV explicit ImageData(const Image& img, int borders)
            : rows(img.getRows()), columns(img.getColumns()),
            appliedBorders(borders), maxGrayLevel(img.getMaxGrayLevel()){};
    // Getters
    CUDA_HOSTDEV unsigned int getRows() const;
    CUDA_HOSTDEV unsigned int getColumns() const;
    CUDA_HOSTDEV unsigned int getMaxGrayLevel() const;
    // Borders applied to the original image
    CUDA_HOSTDEV int getBorderSize() const;
    // Debug method
    CUDA_HOSTDEV void printElements(unsigned int* pixels) const;

private:
    const unsigned int rows;
    const unsigned int columns;
    const unsigned int maxGrayLevel;
    // Amount of borders applied to each side of the original image
    int appliedBorders;
};


#endif /* IMAGEDATA_H_ */
