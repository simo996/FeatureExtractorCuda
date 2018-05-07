//
// Created by simo on 07/05/18.
//

#include <iostream>
#include "GrayPair.h"


// Extract from a pair of gray levels i,g and their multiplicity
struct GrayPair unPack(const int value, const int numberOfPairs, const int maxGrayLevel)
{
    struct GrayPair couple;
    int roundDivision = value / numberOfPairs; // risultato intero
    couple.multiplicity = (value - roundDivision * numberOfPairs)+1;
    couple.grayLevelI = roundDivision / maxGrayLevel; // risultato intero
    couple.grayLevelJ = roundDivision - (maxGrayLevel * couple.grayLevelI);
    return couple;
}

struct AggregatedPair aggregatedPairUnPack(const int value, const int numberOfPairs)
{
    struct AggregatedPair pair;
    int roundDivision = value / numberOfPairs;
    pair.aggregatedGrayLevels = roundDivision; // risultato intero
    pair.multiplicity = value - roundDivision * numberOfPairs ;
    return pair;
}

void printPair(const int value, const int numberOfPairs, const int maxGrayLevel)
{
    GrayPair temp = unPack(value, numberOfPairs, maxGrayLevel);
    std::cout << "Codifica: " << value;
    std::cout << "\ti: "<< temp.grayLevelI;
    std::cout << "\tj: " << temp.grayLevelJ;
    std::cout << "\tmult: " << temp.multiplicity << std::endl;
}

void printAggregatedPair(const int value, const int numberOfPairs, const int maxGrayLevel)
{
    AggregatedPair temp = aggregatedPairUnPack(value, numberOfPairs);
    std::cout << "K: "<< temp.aggregatedGrayLevels;
    std::cout << "\tmult: " << temp.multiplicity << std::endl;
}

