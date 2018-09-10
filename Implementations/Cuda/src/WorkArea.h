#ifndef WORKAREA_H_
#define WORKAREA_H_

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

using namespace std;


/*
 * This class handles memory locations used from GLCM class to generate
 * glcm and the other 4 arrays from which features will be extracted + some
 * useful data to pass to GLCM
 *
 * Memory is malloced externally to this class but pointers are grouped here
*/

class WorkArea {
public:
    CUDA_HOSTDEV WorkArea(int length,
            GrayPair* grayPairs,
            AggregatedGrayPair* summedPairs,
            AggregatedGrayPair* subtractedPairs,
            AggregatedGrayPair* xMarginalPairs,
            AggregatedGrayPair* yMarginalPairs,
            double* out):
            numberOfElements(length), grayPairs(grayPairs), summedPairs(summedPairs),
            subtractedPairs(subtractedPairs), xMarginalPairs(xMarginalPairs),
            yMarginalPairs(yMarginalPairs), output(out){};
    // Get the arrays to initial state so another window can be processed
    CUDA_DEV void cleanup();
    // Invocation of free on the pointers
    CUDA_DEV void release();

    // Where the GLCM will be assembled
    GrayPair* grayPairs;
    // Where the aggregated (sum or diff) representations will be assembled
    AggregatedGrayPair* summedPairs;
    AggregatedGrayPair* subtractedPairs;
    // Where the marginal (x or y) representations will be assembled
    AggregatedGrayPair* xMarginalPairs;
    AggregatedGrayPair* yMarginalPairs;

    // Where to put the feature computed
    double* output;
    // maximum length of all the arrays
    int numberOfElements;

};

#endif /* WORKAREA_H_ */
