
#ifndef FEATUREEXTRACTOR_IMAGE_H
#define FEATUREEXTRACTOR_IMAGE_H

#include "Image.h"

using namespace std;

/*
 * This class embeds metadata about the acquired image:
 * - pysical dimensions (height, width as rows and columns)
 * - the maximum gray level that could be encountered according to its type
 *
 * On the CPU only the "Image" class should be used; it's still present to
 * facilitate development and debugging of the GPU version
*/

class ImageData {
public:
    explicit ImageData(unsigned int rows, unsigned int columns, unsigned int mxGrayLevel)
            : rows(rows), columns(columns), maxGrayLevel(mxGrayLevel){};
    // Strip metadata from the "complete" Image class
    explicit ImageData(const Image& img)
            : rows(img.getRows()), columns(img.getColumns()), maxGrayLevel(img.getMaxGrayLevel()){};
    // Getters
    unsigned int getRows() const;
    unsigned int getColumns() const;
    unsigned int getMaxGrayLevel() const;
    // Debug method
    void printElements(unsigned int* pixels) const;

private:
    const unsigned int rows;
    const unsigned int columns;
    const unsigned int maxGrayLevel;
};


#endif //FEATUREEXTRACTOR_IMAGE_H
