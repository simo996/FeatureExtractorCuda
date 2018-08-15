//
// Created by simone on 14/08/18.
//

#include <cstring>
#include "WorkArea.h"

void WorkArea::cleanup() {
    memset(grayPairs, 0, sizeof(GrayPair) * numberOfElements);
    memset(summedPairs, 0, sizeof(AggregatedGrayPair) * numberOfElements);
    memset(subtractedPairs, 0, sizeof(AggregatedGrayPair) * numberOfElements);
    memset(xMarginalPairs, 0, sizeof(AggregatedGrayPair) * numberOfElements);
    memset(yMarginalPairs, 0, sizeof(AggregatedGrayPair) * numberOfElements);
}