#ifndef PRE_CUDA_IMAGEALLOCATED_H
#define PRE_CUDA_IMAGEALLOCATED_H

#include <iostream>
#include <vector>

using namespace std;

/**
 * This class represent the acquired image; it embeds:
 * - all its pixels as unsigned ints
 * - pysical dimensions (height, width as rows and columns)
 * - the maximum gray level that could be encountered according to its type
*/
class Image {
public:
    /**
     * Constructor of the image
     * @param pixels: all pixels of the image transformed into unsigned int
     * @param rows
     * @param columns
     * @param mxGrayLevel: maximum gray level that can be encountered in the
     * image; depends on the image type and eventual quantitization applied
     */
    Image(vector<unsigned int> pixels, unsigned int rows, unsigned int columns, unsigned int mxGrayLevel)
            :pixels(pixels), rows(rows), columns(columns), maxGrayLevel(mxGrayLevel){};
    /**
     * Getter
     * @return the pixels of the image
     */
    vector<unsigned int> getPixels() const;
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
     * DEBUG METHOD. Print all the pixels of the image
     */
    void printElements() const;

private:
    vector<unsigned int> pixels;
    const unsigned int rows;
    const unsigned int columns;
    const unsigned int maxGrayLevel;
};


#endif //PRE_CUDA_IMAGEALLOCATED_H
