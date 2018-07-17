//
// Created by simo on 11/07/18.
//

#include <iostream>
#include <algorithm> // is it necessary ??
#include <assert.h>
#include "GLCM.h"
#include "GrayPair.h"
#include "AggregatedGrayPair.h"

using namespace std;

// Constructors
GLCM::GLCM(int distance, int shiftRows, int shiftColumns, int windowDimension, int maxGrayLevel, bool simmetric){
    this->distance = distance;
    this->shiftRows = shiftRows;
    this->shiftColumns = shiftColumns;
    this->windowDimension = windowDimension;
    this->simmetric = simmetric;
    this->maxGrayLevel = maxGrayLevel;
    this->numberOfPairs = getBorderRows() * getBorderColumns();
}

int GLCM::getNumberOfPairs() const {
  return numberOfPairs;
}

int GLCM::getMaxGrayLevel() const {
    return maxGrayLevel;
}

int GLCM::getBorderRows() const{
   return (windowDimension - (distance * abs(shiftRows)));
}

int GLCM::getBorderColumns() const{
    return (windowDimension - (distance * abs(shiftColumns)));
}

void GLCM::printGLCMData() const{
    cout << endl;
    cout << "***\tGLCM Data\t***" << endl;
    cout << "Shift rows : " << shiftRows << endl;
    cout << "Shift columns: " << shiftColumns  << endl;
    cout << "Father Window dimension: "<< windowDimension  << endl;
    cout << "Border Rows: "<< getBorderRows()  << endl;
    cout << "Border Columns: " << getBorderColumns()  << endl;
    cout << "Simmetric: ";
    if(simmetric){
        cout << "Yes" << endl;
    }
    else{
        cout << "No" << endl;
    }
    cout << endl;
}

void GLCM::printGLCMElements() const{
    cout << "* GrayPairs *" << endl;

    typedef map<GrayPair, int>::const_iterator MapIterator;
    for(MapIterator mi=grayPairsMap.begin(); mi!=grayPairsMap.end(); mi++)
    {
        mi->first.printPair();
        cout << "\tmult: " <<mi->second;
        cout << endl;
    }
}


int GLCM::computeColumnOffset()
{
    int initialColumnOffset = 0; // for 0°,45°,90°
    if((shiftRows * shiftColumns) > 0) // 135°
        initialColumnOffset = 1;
    return initialColumnOffset;
}

int GLCM::computeRowOffset()
{
    int initialRowOffset = 1; // for 45°,90°,135°
    if((shiftRows == 0) && (shiftColumns > 0))
        initialRowOffset = 0; // for 0°
    return initialRowOffset;
}


void GLCM::initializeElements(const vector<int>& inputPixels) {
    // Define subBorders offset depending on orientation
    int initialColumnOffset = computeColumnOffset();
    int initialRowOffset = computeRowOffset();
                
    int referenceGrayLevel;
    int neighborGrayLevel;
    for (int i = 0; i < getBorderRows() ; i++)
    {
        for (int j = 0; j < getBorderColumns(); j++)
        {
            // Extract the two pixels in the pair
                    // TODO extract addressing function
            int referenceIndex = ((i + initialRowOffset) * windowDimension) + (j + initialColumnOffset);
            referenceGrayLevel = inputPixels[referenceIndex];
            assert(referenceIndex >= 0);
            int neighborIndex = (i * windowDimension) + (j + initialColumnOffset + shiftColumns);
            assert(neighborIndex >= 0);
            neighborGrayLevel = inputPixels[neighborIndex];
            
            GrayPair actualPair(referenceGrayLevel, neighborGrayLevel);
            grayPairsMap[actualPair]+=1;
            if(simmetric) // Create the simmetric counterpart
            {
                GrayPair simmetricPair(neighborGrayLevel, referenceGrayLevel);
                grayPairsMap[simmetricPair]+=1;
            }
            
        }
    }

}

// AGGREGATED representatio for sum and difference
map<AggregatedGrayPair, int> GLCM::codifySummedPairs() const{
    typedef map<GrayPair, int>::const_iterator MapIterator;
    map<AggregatedGrayPair, int> aggregatedPairs;
    for(MapIterator mi=grayPairsMap.begin(); mi!=grayPairsMap.end(); mi++)
    {
        int k= mi->first.getGrayLevelI() + mi->first.getGrayLevelJ();
        AggregatedGrayPair element(k);

        aggregatedPairs[element]+=mi->second;
    }
    return aggregatedPairs;
}

map<AggregatedGrayPair, int> GLCM::codifySubtractedPairs() const{
    typedef map<GrayPair, int>::const_iterator MapIterator;
    map<AggregatedGrayPair, int> aggregatedPairs;
    for(MapIterator mi=grayPairsMap.begin(); mi!=grayPairsMap.end(); mi++)
    {
        int k= abs(mi->first.getGrayLevelI() - mi->first.getGrayLevelJ());
        AggregatedGrayPair element(k);

        aggregatedPairs[element]+=mi->second;
    }
    return aggregatedPairs;
}

void GLCM::printAggregated() const{
    printGLCMAggregatedElements(codifySummedPairs(), true);
    printGLCMAggregatedElements(codifySubtractedPairs(), false);
}

// Metodo statico ?
void GLCM::printGLCMAggregatedElements(map<AggregatedGrayPair, int> input, bool areSummed) const{
    cout << endl;
    if(areSummed)
        cout << "* Summed grayPairsMap *" << endl;
    else
        cout << "* Subtracted grayPairsMap *" << endl;
    typedef map<AggregatedGrayPair, int>::const_iterator MapIterator;
    for(MapIterator mi=input.begin(); mi!=input.end(); mi++)
    {
        mi->first.printPair();
        cout << "\tmult: " << mi->second << endl;
    }
}

// compute marginal frequency f(x)=sum(GrayPair<x, qualunque>)
map<int, int> GLCM::codifyXMarginalProbabilities() const{
    map<int, int> xMarginalPairs;

    typedef map<GrayPair, int>::const_iterator GrayPairsMapIterator;
    for(GrayPairsMapIterator mi=grayPairsMap.begin(); mi!=grayPairsMap.end(); mi++)
    {
        int firstGrayPair = mi->first.getGrayLevelI();
        xMarginalPairs[firstGrayPair] += mi->second;
    }

    printMarginalProbability(xMarginalPairs, 'x');

    return xMarginalPairs;
}

// compute marginal frequency f(y)=sum(GrayPair<qualunque, y>)
map<int, int> GLCM::codifyYMarginalProbabilities() const{
    map<int, int> yMarginalPairs;
    
    typedef map<GrayPair, int>::const_iterator GrayPairsMapIterator;
    for(GrayPairsMapIterator mi=grayPairsMap.begin(); mi!=grayPairsMap.end(); mi++)
    {
        int secondGrayPair= mi->first.getGrayLevelJ();
        yMarginalPairs[secondGrayPair]+=mi->second;
    }

    printMarginalProbability(yMarginalPairs, 'y');

    return yMarginalPairs;
}

void GLCM::printMarginalProbability(const map<int, int> marginalProb, const char symbol) {
    if(symbol == 'x')
            cout << endl << "* xMarginal Codifica" << endl;
        else
            cout << endl << "* yMarginal Codifica" << endl;

    typedef map<int, int>::const_iterator MI;
    for(MI mi=marginalProb.begin(); mi!=marginalProb.end(); mi++)
    {
        if(symbol == 'x')
            cout << "("<< (mi->first)<<", X):\t" << mi->second << endl;
        else
            cout << "(X, "<< (mi->first ) <<"):\t" << mi->second << endl;
    }
}
