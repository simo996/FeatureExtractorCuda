
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
    // Getters
    /**
     * Getter
     * @return the pixels of the image
     */
    vector<unsigned int> getPixels() const; // Pixels must be moved to gpu with a plain pointer
    // Only pysical dimensions can be used in GPU
    /**
    * Getter
    * @return the number of rows of the image
    */
    CUDA_HOSTDEV unsigned int getRows() const;
    /**
    * Getter
    * @return the number of columns of the image
    */
    CUDA_HOSTDEV unsigned int getColumns() const;
    /**
     * @return maximum gray level that can be encountered in the
     * image; depends on the image type and eventual quantitization applied
     */
    CUDA_HOSTDEV unsigned int getMaxGrayLevel() const;
    /**
     * DEBUG METHOD. Print all the pixels of the image
     */
    CUDA_HOST void printElements() const;

private:
    vector<unsigned int> pixels;
    const unsigned int rows;
    const unsigned int columns;
    const unsigned int maxGrayLevel;
};


#endif /* IMAGE_H_ */
