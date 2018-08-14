//
// Created by simone on 12/08/18.
//

#ifndef PRE_CUDA_IMAGEALLOCATED_H
#define PRE_CUDA_IMAGEALLOCATED_H

#include <iostream>
#include <vector>

using namespace std;

class ImageAllocated {
public:
    ImageAllocated(const vector<unsigned int> pixels, unsigned int rows, unsigned int columns, unsigned int mxGrayLevel)
            :pixels(pixels), rows(rows), columns(columns), maxGrayLevel(mxGrayLevel){};
    const vector<unsigned int> getPixels() const;
    unsigned int getRows() const;
    unsigned int getColumns() const;
    unsigned int getMaxGrayLevel() const;
    void printElements() const;

private:
    // Should belong to private class
    const vector<unsigned int> pixels;
    const unsigned int rows;
    const unsigned int columns;
    const unsigned int maxGrayLevel;
};


#endif //PRE_CUDA_IMAGEALLOCATED_H
