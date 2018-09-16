#ifndef GLCM_H_
#define GLCM_H_

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__ 
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif

#include "GrayPair.h"
#include "AggregatedGrayPair.h"
#include "Window.h"
#include "ImageData.h"
#include "WorkArea.h"


using namespace std;

/* This class generates all the elements needed to compute the features
 * from the pixel pairs of the image.
*/

class GLCM {
public:
    // GLCM
    GrayPair* grayPairs;
    int effectiveNumberOfGrayPairs;
    // Array of Pairs (k, frequency) where K is the sum of both gray levels of the pixel pair
    AggregatedGrayPair* summedPairs;
    int numberOfSummedPairs;
    // Array of Pairs (k, frequency) where K is the difference of both gray levels of the pixel pair
    AggregatedGrayPair* subtractedPairs;
    int numberOfSubtractedPairs;
    /* Array of Pairs (k, frequency) where K is the gray level of the reference
     * pixel in the pair */
    AggregatedGrayPair* xMarginalPairs;
    int numberOfxMarginalPairs;
    /* Array of Pairs (k, frequency) where K is the gray level of the neighbor
     * pixel in the pair */
    AggregatedGrayPair* yMarginalPairs;
    int numberOfyMarginalPairs;

    // Standard initializer constructor
    CUDA_DEV GLCM(const unsigned int * pixels, const ImageData& image, Window& windowData, WorkArea& wa);
    CUDA_DEV ~GLCM();

    // Getters method exposed for feature computer class
    CUDA_DEV int getNumberOfPairs() const;
    CUDA_DEV int getMaxGrayLevel() const;

    // Utilities
    CUDA_DEV void printGLCM() const;

private:
   // Pixels of the image
    const unsigned int * pixels;
    // Metadata about the image (dimensions, maxGrayLevel)
    ImageData img;
    // Effective length of the glcm
    int numberOfPairs;
    // Metadata about the window where this GLCM is computed
    WorkArea& workArea;
    // Memory location that will store glcm and other 4 arrays of AggregatedPairs
    Window windowData;

     // Additional shifts applied reflecting the direction that is being computed
    CUDA_DEV int computeWindowColumnOffset();
    CUDA_DEV int computeWindowRowOffset();
    // Geometric limits in the windows where this GLCM is computed
    CUDA_DEV int getWindowRowsBorder() const;
    CUDA_DEV int getWindowColsBorder() const;
    // Addressing methods to get pixels in the pair
    CUDA_DEV int getReferenceIndex(int i, int j, int initialRowOffset, int initialColumnOffset);
    CUDA_DEV int getNeighborIndex(int i, int j, int initialColumnOffset);
    // Methods to build the glcm from input pixel and directional data
    CUDA_DEV void insertElement(GrayPair* elements, GrayPair actualPair,
            uint& lastInsertionPosition);
    CUDA_DEV void insertElement(AggregatedGrayPair* elements,
            AggregatedGrayPair actualPair, uint& lastInsertionPosition);
    CUDA_DEV void initializeGlcmElements();
    // Representations useful for aggregated features
    CUDA_DEV void codifyAggregatedPairs();
    // Representation useful for HXY
    CUDA_DEV void codifyMarginalPairs() ;

    // debug printing methods
    CUDA_DEV void printGLCMData() const;
    CUDA_DEV void printGLCMElements() const;
    CUDA_DEV void printAggregated() const;
    CUDA_DEV void printMarginalProbabilityElements() const;
    CUDA_DEV void printGLCMAggregatedElements(bool areSummed) const;
};
#endif /* GLCM_H_ */
