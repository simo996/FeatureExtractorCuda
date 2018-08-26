/*
 * This class embeds pixels and image's metadata used by other components
*/

#ifndef FEATUREEXTRACTOR_IMAGE_H
#define FEATUREEXTRACTOR_IMAGE_H

#include "Image.h"

using namespace std;
class ImageData {
public:
    explicit ImageData(unsigned int rows, unsigned int columns, unsigned int mxGrayLevel)
            : rows(rows), columns(columns), maxGrayLevel(mxGrayLevel){};
    explicit ImageData(const Image& img)
            : rows(img.getRows()), columns(img.getColumns()), maxGrayLevel(img.getMaxGrayLevel()){};
    unsigned int getRows() const;
    unsigned int getColumns() const;
    unsigned int getMaxGrayLevel() const;
    void printElements(unsigned int* pixels) const;

private:
    const unsigned int rows;
    const unsigned int columns;
    const unsigned int maxGrayLevel;
};


#endif //FEATUREEXTRACTOR_IMAGE_H
