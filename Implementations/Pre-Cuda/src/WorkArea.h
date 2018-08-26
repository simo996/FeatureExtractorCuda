//
// Created by simone on 14/08/18.
//

#ifndef PRE_CUDA_WORKAREA_H
#define PRE_CUDA_WORKAREA_H

#include <vector>
#include "GrayPair.h"
#include "AggregatedGrayPair.h"

using namespace std;

class WorkArea {
public:
    WorkArea(int length,
            vector<GrayPair>& grayPairs,
            vector<AggregatedGrayPair>& summedPairs,
            vector<AggregatedGrayPair>& subtractedPairs,
            vector<AggregatedGrayPair>& xMarginalPairs,
            vector<AggregatedGrayPair>& yMarginalPairs,
            vector<double>& out):
            numberOfElements(length), grayPairs(grayPairs.data()), summedPairs(summedPairs.data()),
            subtractedPairs(subtractedPairs.data()), xMarginalPairs(xMarginalPairs.data()),
            yMarginalPairs(yMarginalPairs.data()), output(out.data()){};
    void cleanup();
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


#endif //PRE_CUDA_WORKAREA_H
