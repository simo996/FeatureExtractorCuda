#ifndef PRE_CUDA_IMAGEALLOCATED_H
#define PRE_CUDA_IMAGEALLOCATED_H

#include <iostream>
#include <vector>

using namespace std;

/* This class represent the acquired image; it embeds:
 * - all its pixels as unsigned ints
 * - pysical dimensions (height, width as rows and columns)
 * - the maximum gray level that could be encountered according to its type
*/
class Image {
public:
    Image(vector<unsigned int> pixels, unsigned int rows, unsigned int columns, unsigned int mxGrayLevel)
            :pixels(pixels), rows(rows), columns(columns), maxGrayLevel(mxGrayLevel){};
    // Getters
    vector<unsigned int> getPixels() const;
    unsigned int getRows() const;
    unsigned int getColumns() const;
    unsigned int getMaxGrayLevel() const;
    // Debug method
    void printElements() const;

private:
    vector<unsigned int> pixels;
    const unsigned int rows;
    const unsigned int columns;
    const unsigned int maxGrayLevel;
};


#endif //PRE_CUDA_IMAGEALLOCATED_H
