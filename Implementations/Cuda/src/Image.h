
#ifndef IMAGE_H_
#define IMAGE_H_

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__ 
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif

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
    // Pixels will be used in GPU threads through a special handling
    vector<unsigned int> getPixels() const;
    // Only pysical dimension can be used in GPU
    CUDA_HOSTDEV unsigned int getRows() const;
    CUDA_HOSTDEV unsigned int getColumns() const;
    CUDA_HOSTDEV unsigned int getMaxGrayLevel() const;
    // Debug method
    CUDA_HOST void printElements() const;

private:
    // Should belong to private class
    vector<unsigned int> pixels;
    const unsigned int rows;
    const unsigned int columns;
    const unsigned int maxGrayLevel;
};


#endif /* IMAGE_H_ */
