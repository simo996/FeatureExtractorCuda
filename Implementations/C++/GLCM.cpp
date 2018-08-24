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
GLCM::GLCM(const Image& image, Window& windowData)
        : img(image), windowData(windowData){
    this->numberOfPairs = getWindowRowBorder() * getWindowColsBorder();
    initializeGlcmElements();
    codifySummedPairs();
    codifySubtractedPairs();
    codifyYMarginalProbabilities();
    codifyXMarginalProbabilities();
}

int GLCM::getNumberOfPairs() const {
    if(windowData.symmetric)
        // Each element was counted twice
        return (2 * numberOfPairs);
    else
        return numberOfPairs;
}

int GLCM::getMaxGrayLevel() const {
    return img.getMaxGrayLevel();
}

int GLCM::getWindowRowBorder() const{
   return (windowData.side - (windowData.distance * abs(windowData.shiftRows)));
}

int GLCM::getWindowColsBorder() const{
    return (windowData.side - (windowData.distance * abs(windowData.shiftColumns)));
}

void GLCM::printGLCMData() const{
    cout << endl;
    cout << "***\tGLCM Data\t***" << endl;
    cout << "Shift rows : " << windowData.shiftRows << endl;
    cout << "Shift columns: " << windowData.shiftColumns  << endl;
    cout << "Father Window side: "<< windowData.side  << endl;
    cout << "Border Rows: "<< getWindowRowBorder()  << endl;
    cout << "Border Columns: " << getWindowColsBorder()  << endl;
    cout << "Simmetric: ";
    if(windowData.symmetric){
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
    for(MapIterator mi=grayPairs.begin(); mi!=grayPairs.end(); mi++)
    {
        mi->first.printPair();
        cout << "\tmult: " <<mi->second;
        cout << endl;
    }
}

/*
    columnOffset is a shift value used for reading the correct batch of elements
    from given linearized input pixels; for 135° the first d (distance) elements 
    need to be ignored
*/
inline int GLCM::computeWindowColumnOffset()
{
    int initialColumnOffset = 0; // for 0°,45°,90°
    if((windowData.shiftRows * windowData.shiftColumns) > 0) // 135°
        initialColumnOffset = 1;
    return initialColumnOffset;
}

/*
    rowOffset is a shift value used for reading the correct batch of elements
    from given linearized input pixels according to the direction in use; 
    45/90/135° must skip d (distance) "rows"
*/
inline int GLCM::computeWindowRowOffset()
{
    int initialRowOffset = 1; // for 45°,90°,135°
    if((windowData.shiftRows == 0) && (windowData.shiftColumns > 0))
        initialRowOffset = 0; // for 0°
    return initialRowOffset;
}

// addressing method for reference pixel; see documentation
inline int GLCM::getReferenceIndex(const int i, const int j,
                                   const int initialWindowRowOffset, const int initialWindowColumnOffset){
    int index = (((i + windowData.imageRowsOffset) + (initialWindowRowOffset * windowData.distance)) * img.getRows())
            + ((j + windowData.imageColumnsOffset) + (initialWindowColumnOffset * windowData.distance));
    assert(index >= 0);
    return index;
}

// addressing method for neighbor pixel; see documentation
inline int GLCM::getNeighborIndex(const int i, const int j,
                                  const int initialWindowColumnOffset){
    int index = ((i + windowData.imageRowsOffset) * img.getColumns()) +
            ((j + windowData.imageColumnsOffset) + (initialWindowColumnOffset * windowData.distance) + (windowData.shiftColumns * windowData.distance));
    assert(index >= 0);
    return index;
}
/*
    This method puts inside the map of elements map<GrayPair, int> each
    frequency associated with each pair of grayLevels
*/
void GLCM::initializeGlcmElements() {
    // Define subBorders offset depending on orientation
    int initialWindowColumnOffset = computeWindowColumnOffset();
    int initialWindowRowOffset = computeWindowRowOffset();
    // Offset to locate the starting point of the window inside the sliding image

    uint referenceGrayLevel;
    uint neighborGrayLevel;
    for (int i = 0; i < getWindowRowBorder() ; i++)
    {
        for (int j = 0; j < getWindowColsBorder(); j++)
        {
            // Extract the two pixels in the pair
            int referenceIndex = getReferenceIndex(i, j,
                    initialWindowRowOffset, initialWindowColumnOffset);
            referenceGrayLevel = img.getPixels()[referenceIndex];
            int neighborIndex = getNeighborIndex(i, j,
                    initialWindowColumnOffset);
            neighborGrayLevel = img.getPixels()[neighborIndex];

            GrayPair actualPair(referenceGrayLevel, neighborGrayLevel);
            grayPairs[actualPair] += 1;
            if(windowData.symmetric) // Create the symmetric counterpart
            {
                GrayPair simmetricPair(neighborGrayLevel, referenceGrayLevel);
                grayPairs[simmetricPair] += 1;
            }

        }
    }

}

/*
    This method, given the map<GrayPair, int freq> will produce 
    map<int k, int freq> where k is the sum of both grayLevels of the GrayPair.
    This representation is used in computeSumXXX() features
*/
void GLCM::codifySummedPairs() {
    typedef map<GrayPair, int>::const_iterator MapIterator;

    for(MapIterator mi=grayPairs.begin(); mi!=grayPairs.end(); mi++)
    {
        uint k= mi->first.getGrayLevelI() + mi->first.getGrayLevelJ();
        AggregatedGrayPair element(k);

        summedPairs[element] += mi->second;
    }
}

/*
    This method, given the map<GrayPair, int freq> will produce 
    map<int k, int freq> where k is the absolute difference of both grayLevels
    of the GrayPair.
    This representation is used in computeDiffXXX() features
*/
void GLCM::codifySubtractedPairs(){
    typedef map<GrayPair, int>::const_iterator MapIterator;

    for(MapIterator mi=grayPairs.begin(); mi!=grayPairs.end(); mi++)
    {
        int diff = mi->first.getGrayLevelI() - mi->first.getGrayLevelJ();
        uint k= static_cast<uint>(abs(diff));
        AggregatedGrayPair element(k);

        subtractedPairs[element] += mi->second;
    }
}

void GLCM::printAggregated() const{
    printGLCMAggregatedElements(summedPairs, true);
    printGLCMAggregatedElements(subtractedPairs, false);
}

void GLCM::printGLCMAggregatedElements(map<AggregatedGrayPair, int> input, bool areSummed) const{
    cout << endl;
    if(areSummed)
        cout << "* Summed grayPairsMap *" << endl;
    else
        cout << "* Subtracted grayPairsMap *" << endl;
    typedef map<AggregatedGrayPair, int>::const_iterator MapIterator;
    for(MapIterator mi = input.begin(); mi != input.end(); mi++)
    {
        mi->first.printPair();
        cout << "\tmult: " << mi->second << endl;
    }
}

/*
    This method, given the map<GrayPair, int freq> will produce 
    map<int k, int freq> where k is the REFERENCE grayLevel of the GrayPair 
    while freq is the "marginal" frequency of that level 
    (ie. how many times k is present in all GrayPair<k, ?>)
    This representation is used for computing features HX, HXY, HXY1, imoc
*/
void GLCM::codifyXMarginalProbabilities() {
    typedef map<GrayPair, int>::const_iterator GrayPairsMapIterator;

    for(GrayPairsMapIterator mi=grayPairs.begin(); mi!=grayPairs.end(); mi++)
    {
        uint firstGrayPair = mi->first.getGrayLevelI();
        xMarginalPairs[firstGrayPair] += mi->second;
    }
}

/*
    This method, given the map<GrayPair, int freq> will produce 
    map<int k, int freq> where k is the NEIGHBOR grayLevel of the GrayPair 
    while freq is the "marginal" frequency of that level 
    (ie. how many times k is present in all GrayPair<?, k>)
    This representation is used for computing features HX, HXY, HXY1, imoc
*/
void GLCM::codifyYMarginalProbabilities() {
    typedef map<GrayPair, int>::const_iterator GrayPairsMapIterator;

    for(GrayPairsMapIterator mi=grayPairs.begin(); mi!=grayPairs.end(); mi++)
    {
        uint secondGrayPair = mi->first.getGrayLevelJ();
        yMarginalPairs[secondGrayPair] += mi->second;
    }
}

void GLCM::printMarginalProbability(const map<uint, int> marginalProb, const char symbol) {
    if(symbol == 'x')
            cout << endl << "* xMarginal Codifica" << endl;
        else
            cout << endl << "* yMarginal Codifica" << endl;

    typedef map<uint, int>::const_iterator MI;
    for(MI mi=marginalProb.begin(); mi!=marginalProb.end(); mi++)
    {
        if(symbol == 'x')
            cout << "("<< (mi->first)<<", X):\t" << mi->second << endl;
        else
            cout << "(X, "<< (mi->first ) <<"):\t" << mi->second << endl;
    }
}
