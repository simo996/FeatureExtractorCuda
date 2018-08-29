/*
 * WorkArea.cpp
 *
 *  Created on: 26/ago/2018
 *      Author: simone
 */

#include "WorkArea.h"

__device__ void WorkArea::cleanup() {
    GrayPair voidElement; // 0 in each field
    AggregatedGrayPair voidAggregatedElement; // 0 in each field
    for (int i = 0; i < numberOfElements; ++i) {
        grayPairs[i] = voidElement;
        summedPairs[i] = voidAggregatedElement;
        subtractedPairs[i] = voidAggregatedElement;
        xMarginalPairs[i] = voidAggregatedElement;
        yMarginalPairs[i] = voidAggregatedElement;
    }
}

__device__ void WorkArea::release(){
	free(grayPairs);
    free(summedPairs);
    free(subtractedPairs);
    free(xMarginalPairs);
    free(yMarginalPairs);
}