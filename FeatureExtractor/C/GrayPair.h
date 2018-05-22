//
// Created by simo on 07/05/18.
//

#ifndef FEATURESEXTRACTOR_GRAYPAIR_H
#define FEATURESEXTRACTOR_GRAYPAIR_H

struct GrayPair{
	int grayLevelI;
	int grayLevelJ;
	int multiplicity;
};

struct AggregatedPair{
	int aggregatedGrayLevels; //i+j or abs(i-j)
	int multiplicity;
};

struct GrayPair unPack(const int value, const int numberOfPairs, const int maxGrayLevel);
struct AggregatedPair aggregatedPairUnPack(const int value, const int numberOfPairs);

bool compareEqualsGrayPairs(const struct GrayPair first, const struct GrayPair second);
bool compareEqualsGrayPairs(const int first, const int second, const int numberOfPairs, const int maxGrayLevel);
bool compareEqualsAggregatedPairs(const struct AggregatedPair first, const struct AggregatedPair second);
bool compareEqualsAggregatedPairs(const int first, const int second, const int numberOfPairs);

void printPair(const int value, const int numberOfPairs, const int maxGrayLevel);
void printPair(const struct GrayPair pair, const int numberOfPairs, const int maxGrayLevel);
void printAggregatedPair(const int value, const int numberOfPairs);
void printAggregatedPair(const struct AggregatedPair pair, const int numberOfPairs);

#endif //FEATURESEXTRACTOR_GRAYPAIR_H
