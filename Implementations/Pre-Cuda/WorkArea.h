//
// Created by simone on 14/08/18.
//

#ifndef PRE_CUDA_WORKAREA_H
#define PRE_CUDA_WORKAREA_H


#include "GrayPair.h"
#include "AggregatedGrayPair.h"

class WorkArea {
public:
    WorkArea(int length, GrayPair* pairs, AggregatedGrayPair * summed,
            AggregatedGrayPair * subtracted, AggregatedGrayPair * xMarginal,
            AggregatedGrayPair * yMarginal): numberOfElements(length), grayPairs(pairs), summedPairs(summed),
            subtractedPairs(subtracted), xMarginalPairs(xMarginal), yMarginalPairs(yMarginal){};

    // Set to default (0 for every field) each element
    void cleanup();
    GrayPair * getGrayPairs(){
        return grayPairs;
    };
    AggregatedGrayPair * getSummedPairs(){
        return summedPairs;
    }
    AggregatedGrayPair * getSubtractedPairs(){
        return summedPairs;
    }
    AggregatedGrayPair * getxMarginalPairs(){
        return xMarginalPairs;
    }
    AggregatedGrayPair * getyMarginalPairs(){
        return yMarginalPairs;
    }
private:
    GrayPair * grayPairs;
    AggregatedGrayPair * summedPairs;
    AggregatedGrayPair * subtractedPairs;
    AggregatedGrayPair * xMarginalPairs;
    AggregatedGrayPair * yMarginalPairs;
    int numberOfElements;

};


#endif //PRE_CUDA_WORKAREA_H
