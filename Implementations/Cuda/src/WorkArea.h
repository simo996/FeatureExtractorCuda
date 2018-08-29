/*
 * WorkArea.h
 *
 *  Created on: 26/ago/2018
 *      Author: simone
 */

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
    CUDA_DEV void cleanup();
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
    int numberOfElements;

};

#endif /* WORKAREA_H_ */
