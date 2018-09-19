
#ifndef FEATUREEXTRACTOR_IMAGE_H
#define FEATUREEXTRACTOR_IMAGE_H

#include "Image.h"

using namespace std;

/**
 * This class embeds metadata about the acquired image:
 * - pysical dimensions (height, width as rows and columns)
 * - the maximum gray level that could be encountered according to its type
 *
 * On the CPU only the "Image" class should be used; it's still present to
 * facilitate development and debugging of the GPU version
*/

class ImageData {
public:
    /**
     * USE THE OTHER CONSTRUCTOR
     * @param rows
     * @param columns
     * @param borders
     * @param mxGrayLevel
     */
    explicit ImageData(unsigned int rows, unsigned int columns, int borders,
            unsigned int mxGrayLevel)
            : rows(rows), columns(columns), appliedBorders(borders),
            maxGrayLevel(mxGrayLevel){};
     /**
      * Constructor that strips metadata from the "complete" Image class
      * @param img complete image with pixels + metadata
      * @param borders applied to each side the original image
      */
    explicit ImageData(const Image& img, int borders)
            : rows(img.getRows()), columns(img.getColumns()),
            appliedBorders(borders), maxGrayLevel(img.getMaxGrayLevel()){};
    // Getters
    /**
     * Getter
     * @return the number of rows of the image
     */
    unsigned int getRows() const;
    /**
     * Getter
     * @return the number of columns of the image
     */
    unsigned int getColumns() const;
    /**
     * @return maximum gray level that can be encountered in the
     * image; depends on the image type and eventual quantitization applied
     */
    unsigned int getMaxGrayLevel() const;
    /**
     *
     * @return borders applied to each side of the original image
     */
    int getBorderSize() const;

private:
    const unsigned int rows;
    const unsigned int columns;
    const unsigned int maxGrayLevel;
    // Amount of borders applied to each side of the original image
    int appliedBorders;
};


#endif //FEATUREEXTRACTOR_IMAGE_H
