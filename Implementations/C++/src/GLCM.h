#ifndef FEATUREEXTRACTOR_GLCM_H
#define FEATUREEXTRACTOR_GLCM_H

#include "GrayPair.h"
#include "AggregatedGrayPair.h"
#include "Window.h"
#include "ImageData.h"
#include "WorkArea.h"


using namespace std;

/**
 * This class generates all the elements needed to compute the features
 * from the pixel pairs of the image.
 */
class GLCM {
public:
    /**
     * GLCM: it contains all the gray pairs found in the window of interest
     */
    GrayPair* grayPairs;
    /**
     * How many different gray pairs were found in the window of interest.
     * It is necessary since the grayPairs array is pre-allocated with the
     * worst case number of elements
     */
    int effectiveNumberOfGrayPairs;
    /**
     * Array of Pairs (k, frequency) where K is the sum of both gray levels
     * of the pixel pair
     */
    AggregatedGrayPair* summedPairs;
    /**
     * How many different added gray pairs were found in the window of
     * interest.
     * It is necessary since the grayPairs array is pre-allocated with the
     * worst case number of elements
     */
    int numberOfSummedPairs;
    /**
     * Array of Pairs (k, frequency) where K is the difference of both gray levels of the pixel pair
     */
    AggregatedGrayPair* subtractedPairs;
    /**
     * How many different subtracted gray pairs were found in the window of
     * interest.
     * It is necessary since the grayPairs array is pre-allocated with the
     * worst case number of elements
     */
    int numberOfSubtractedPairs;
    /**
     * Array of Pairs (k, frequency) where K is the gray level of the reference
     * pixel in the pair
     */
    AggregatedGrayPair* xMarginalPairs;
    /**
    * How many different x-marginal gray pairs were found in the window of
    * interest.
    * It is necessary since the grayPairs array is pre-allocated with the
    * worst case number of elements
    */
    int numberOfxMarginalPairs;
    /**
     * Array of Pairs (k, frequency) where K is the gray level of the neighbor
     * pixel in the pair
     */
    AggregatedGrayPair* yMarginalPairs;
    /**
    * How many different y-marginal gray pairs were found in the window of
    * interest.
    * It is necessary since the grayPairs array is pre-allocated with the
    * worst case number of elements
    */
    int numberOfyMarginalPairs;

     /**
      * Constructor of the GLCM that will also launch the methods to generate
      * all the elements needed for extracting the features from them
      * @param pixels: of the entire image
      * @param image: metadata about the image (physical dimensions,
      * maxGrayLevel, borders)
      * @param windowData: metadata about this window of interest from which
      * the glcm is computed
      * @param wa: memory location where this object will create the arrays of
     * representation needed for computing its features
      */
    GLCM(const unsigned int * pixels, const ImageData& image, Window& windowData, WorkArea& wa);
    ~GLCM();

    // Getters method exposed for feature computer class
    /**
     * Getter. Returns the number of pairs that belongs to the GLCM; this value
     * is used for computing the probability from the frequency of each item
     * @return the number of pairs that belongs to the GLCM
     */
    int getNumberOfPairs() const;
    /**
     * Getter. Returns the maximum grayLevel of the image
     * @return
     */
    int getMaxGrayLevel() const;

    /**
     * DEBUG METHOD. Prints a complete representation of the GLCM object and
     * every one of its array of grayPairs / aggregatedGrayPairs
     */
    void printGLCM() const;

private:
    /**
      * Pixels of the image
      */
    const unsigned int * pixels;
    /**
     * Metadata about the image (dimensions, maxGrayLevel)
     */
    ImageData image;
    /**
     * Window of interest where the glcm is computed
     */
    Window windowData;
    /**
     * Memory location used for computing this window's feature
     */
    WorkArea& workArea;

    /**
     * number of pairs that belongs to the GLCM
     */
    int numberOfPairs;

    /**
     * Compute the shift to apply at the column for locating the pixels of each
     * pair of the glcm; it affects only 135° orientation
     * @return d (distance) pixels need to be ignored
     */
    int computeWindowColumnOffset();
    /**
    * Compute the shift to apply at the row for locating the pixels of each
    * pair of the glcm; it doesn't affect only 0° orientation
    * @return d (distance) pixels need to be ignored
    */
    int computeWindowRowOffset();
    /**
     * Geometric limit of the sub-window
     * @return how many rows of the window need to be considered
     */
    int getWindowRowsBorder() const;
    /**
    * Geometric limit of the sub-window
    * @return how many columns of the window need to be considered
    */
    int getWindowColsBorder() const;
    /**
     * Addressing methods to get the reference pixel in each pair of the glcm
     * @param row in the sub-window of the reference pixel
     * @param col in the sub-window of the reference pixel
     * @param initialRowOffset see computeWindowRowOffset
     * @param initialColumnOffset see computeWindowColOffset
     * @return the index of the pixel in the array of pixels (linearized) of
     * the window
     */
    int getReferenceIndex(int row, int col, int initialRowOffset, int initialColumnOffset);
    /**
     * Addressing methods to get the neighbor pixel in each pair of the glcm
     * @param row in the sub-window of the neighbor pixel
     * @param col in the sub-window of the neighbor pixel
     * @param initialColumnOffset see computeWindowColOffset
     * @return the index of the pixel in the array of pixels (linearized) of
     * the window
     */
    int getNeighborIndex(int row, int col, int initialColumnOffset);
    /**
     * Method that inserts a GrayPair in the pre-allocated memory
     * Uses that convention that GrayPair ( i=0, j=0, frequency=0) means
     * available memory
     */
    void insertElement(GrayPair* elements, GrayPair actualPair,
            uint& lastInsertionPosition);
    /**
     * Method that inserts a AggregatedGrayPair in the pre-allocated memory
     * Uses that convention that AggregateGrayPair (k=0, frequency=0) means
     * available memory
     */
    void insertElement(AggregatedGrayPair* elements,
            AggregatedGrayPair actualPair, uint& lastInsertionPosition);
    /**
     * This method creates array of GrayPairs
     */
    void initializeGlcmElements();
    /**
     * This method will produce the 2 arrays of AggregatedPairs (k, frequency)
     * where k is the sum or difference of both grayLevels of 1 GrayPair.
     * This representation is used in computeSumXXX() and computeDiffXXX() features
     */
    void codifyAggregatedPairs();
    /**
     * This method will produce the 2 arrays of AggregatedPairs (k, frequency)
     * where k is one grayLevel of GLCM and frequency is the "marginal" frequency of that level
     * (ie. how many times k is present in all GrayPair<k, ?>)
     * This representation is used for computing features HX, HXY, HXY1, imoc
     */
    void codifyMarginalPairs() ;

    // debug printing methods
    /**
     * DEBUG METHOD. Prints a complete representation of the GCLM class and
     * the content of all its arrays of elements
     */
    void printGLCMData() const;
    /**
     * DEBUG METHOD. Prints all the grayPairs of the GLCM
     */
    void printGLCMElements() const;
    /**
     * DEBUG METHOD. Prints all the summedGrayPairs and subtractedGrayPairs
     */
    void printAggregated() const;
    /**
     * DEBUG METHOD. Prints all the xmarginalGrayPairs and ymarginalGrayPairs
     */
    void printMarginalProbabilityElements() const;
    void printGLCMAggregatedElements(bool areSummed) const;

};


#endif //FEATUREEXTRACTOR_GLCM_H
