#ifndef FEATUREEXTRACTOR_GLCM_H
#define FEATUREEXTRACTOR_GLCM_H

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
    GLCM(const unsigned int * pixels, const ImageData& image, Window& windowData, WorkArea& wa);
    ~GLCM();

    // Getters method exposed for feature computer class
    int getNumberOfPairs() const;
    int getMaxGrayLevel() const;

    // Utilities
    void printGLCM() const;

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
    int computeWindowColumnOffset();
    int computeWindowRowOffset();
    // Geometric limits in the windows where this GLCM is computed
    int getWindowRowsBorder() const;
    int getWindowColsBorder() const;
    // Addressing methods to get pixels in the pair
    int getReferenceIndex(int i, int j, int initialRowOffset, int initialColumnOffset);
    int getNeighborIndex(int i, int j, int initialColumnOffset);
    // Methods to build the glcm from input pixel and directional data
    void insertElement(GrayPair* elements, GrayPair actualPair,
            uint& lastInsertionPosition);
    void insertElement(AggregatedGrayPair* elements,
            AggregatedGrayPair actualPair, uint& lastInsertionPosition);
    void initializeGlcmElements();
    // Representations useful for aggregated features
    void codifyAggregatedPairs();
    // Representation useful for HXY
    void codifyMarginalPairs() ;

    // debug printing methods
    void printGLCMData() const;
    void printGLCMElements() const;
    void printAggregated() const;
    void printMarginalProbabilityElements() const;
    void printGLCMAggregatedElements(bool areSummed) const;

};


#endif //FEATUREEXTRACTOR_GLCM_H
