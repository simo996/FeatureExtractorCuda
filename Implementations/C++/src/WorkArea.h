#ifndef PRE_CUDA_WORKAREA_H
#define PRE_CUDA_WORKAREA_H

#include "GrayPair.h"
#include "AggregatedGrayPair.h"

using namespace std;

/**
 * This class handles memory locations used from GLCM class to generate
 * glcm and the other 4 arrays from which features will be extracted + some
 * useful data to pass to GLCM
 *
 * Memory is malloced externally to this class but pointers are grouped here
*/

class WorkArea {
public:
    /**
     * Initialization
     * @param length: number of pairs of each window
     * @param grayPairs: memory space where the array of GrayPairs is created
     * for each window of the image
     * @param summedPairs: memory space where the array of summedGrayPairs
     * is created for each window of the image
     * @param subtractedPairs: memory space where the array of subtractedGrayPairs
     * is created for each window of the image
     * @param xMarginalPairs: memory space where the array of x-marginalGrayPairs
     * is created for each window of the image
     * @param yMarginalPairs: memory space where the array of y-marginalGrayPairs
     * is created for each window of the image
     * @param out: memory space where all the features values will be put
     */
    WorkArea(int length,
            GrayPair* grayPairs,
            AggregatedGrayPair* summedPairs,
            AggregatedGrayPair* subtractedPairs,
            AggregatedGrayPair* xMarginalPairs,
            AggregatedGrayPair* yMarginalPairs,
            double* out):
            numberOfElements(length), grayPairs(grayPairs), summedPairs(summedPairs),
            subtractedPairs(subtractedPairs), xMarginalPairs(xMarginalPairs),
            yMarginalPairs(yMarginalPairs), output(out){};
    /**
     * Get the arrays to initial state so another window can be processed
     */
    void cleanup();
    /**
     * Invocation of free on the pointers of all the meta-Arrays of pairs
     */
    void release();
    /**
     * Where the GLCM will be assembled
     */
    GrayPair* grayPairs;
    /**
     * Where the sum-aggregated representations will be assembled
     */
    AggregatedGrayPair* summedPairs;
    /**
    * Where the diff-aggregated representations will be assembled
    */
    AggregatedGrayPair* subtractedPairs;
    /**
    * Where the x-marginalPairs representations will be assembled
    */
    AggregatedGrayPair* xMarginalPairs;
    /**
   * Where the y-marginalPairs representations will be assembled
   */
    AggregatedGrayPair* yMarginalPairs;
    /**
     * memory space where all the features values will be put
     */
    double* output;
    /**
     * number of pairs of each window
     */
    int numberOfElements;

};


#endif //PRE_CUDA_WORKAREA_H
