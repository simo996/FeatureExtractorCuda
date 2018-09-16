
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
    explicit ImageData(unsigned int rows, unsigned int columns, int borders,
            unsigned int mxGrayLevel)
            : rows(rows), columns(columns), appliedBorders(borders),
            maxGrayLevel(mxGrayLevel){};
    // Strip metadata from the "complete" Image class
    explicit ImageData(const Image& img, int borders)
            : rows(img.getRows()), columns(img.getColumns()),
            appliedBorders(borders), maxGrayLevel(img.getMaxGrayLevel()){};
    // Getters
    unsigned int getRows() const;
    unsigned int getColumns() const;
    unsigned int getMaxGrayLevel() const;
    // Borders applied to the original image
    int getBorderSize() const;
    // Debug method
    void printElements(unsigned int* pixels) const;

private:
    const unsigned int rows;
    const unsigned int columns;
    const unsigned int maxGrayLevel;
    // Amount of borders applied to each side of the original image
    int appliedBorders;
};


#endif //FEATUREEXTRACTOR_IMAGE_H
