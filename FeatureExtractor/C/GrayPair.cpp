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
	pair.multiplicity = value - (roundDivision * numberOfPairs) +1;
	return pair;
}


bool compareEqualsGrayPairs(const struct GrayPair first, const struct GrayPair second)
{
	if((first.grayLevelI == second.grayLevelI) && (first.grayLevelJ == second.grayLevelJ))
		return true;
	else
		return false;
}

bool compareEqualsGrayPairs(const int first, const int second, const int numberOfPairs, const int maxGrayLevel)
{
	GrayPair a, b;
	a = unPack(first, numberOfPairs, maxGrayLevel);
	b = unPack(second, numberOfPairs, maxGrayLevel);
	if((a.grayLevelI == b.grayLevelI) && (a.grayLevelJ == b.grayLevelJ))
		return true;
	else
		return false;
}

bool compareEqualsAggregatedPairs(const struct AggregatedPair first, const struct AggregatedPair second)
{
    if(first.aggregatedGrayLevels == second.aggregatedGrayLevels)
        return true;
    else
        return false;
}

bool compareEqualsAggregatedPairs(const int first, const int second, const int numberOfPairs)
{
    AggregatedPair a, b;
    a = aggregatedPairUnPack(first, numberOfPairs);
    b = aggregatedPairUnPack(second, numberOfPairs);
    if(a.aggregatedGrayLevels == b.aggregatedGrayLevels)
        return true;
    else
        return false;
}

void printPair(const int value, const int numberOfPairs, const int maxGrayLevel)
{
	GrayPair temp = unPack(value, numberOfPairs, maxGrayLevel);
	std::cout << "Codifica: " << value;
	std::cout << "\ti: "<< temp.grayLevelI;
	std::cout << "\tj: " << temp.grayLevelJ;
	std::cout << "\tmult: " << temp.multiplicity << std::endl;
}

void printPair(const struct GrayPair pair, const int numberOfPairs, const int maxGrayLevel)
{
	std::cout << "Codifica: " << (((pair.grayLevelI * maxGrayLevel + pair.grayLevelJ) * numberOfPairs) + pair.multiplicity-1);
	std::cout << "\ti: "<< pair.grayLevelI;
	std::cout << "\tj: " << pair.grayLevelJ;
	std::cout << "\tmult: " << pair.multiplicity << std::endl;
}

void printAggregatedPair(const int value, const int numberOfPairs)
{
	AggregatedPair temp = aggregatedPairUnPack(value, numberOfPairs);
	std::cout << "Codifica: "<< value;
	std::cout << "\tK: "<< temp.aggregatedGrayLevels;
	std::cout << "\tmult: " << temp.multiplicity << std::endl;
}

void printAggregatedPair(const struct AggregatedPair pair, const int numberOfPairs)
{
	std::cout << "Codifica: "<< (pair.aggregatedGrayLevels * numberOfPairs) + pair.multiplicity + 1;
	std::cout << "\tK: "<< pair.aggregatedGrayLevels;
	std::cout << "\tmult: " << pair.multiplicity << std::endl;
}