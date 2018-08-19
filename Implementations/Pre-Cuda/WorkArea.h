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
    WorkArea(int length, vector<GrayPair>& grayPairs, vector<AggregatedGrayPair>& summedPairs,
            vector<AggregatedGrayPair>& subtractedPairs, vector<AggregatedGrayPair>& xMarginalPairs,
            vector<AggregatedGrayPair>& yMarginalPairs, vector<vector<vector<double>>>& out):
            numberOfElements(length), grayPairs(grayPairs), summedPairs(summedPairs),
            subtractedPairs(subtractedPairs), xMarginalPairs(xMarginalPairs),
            yMarginalPairs(yMarginalPairs), output(out){};
    void cleanup();

    vector<GrayPair>& grayPairs;
    vector<AggregatedGrayPair>& summedPairs;
    vector<AggregatedGrayPair>& subtractedPairs;
    vector<AggregatedGrayPair>& xMarginalPairs;
    vector<AggregatedGrayPair>& yMarginalPairs;
    vector<vector<vector<double>>>& output;
    int numberOfElements;

};


#endif //PRE_CUDA_WORKAREA_H
