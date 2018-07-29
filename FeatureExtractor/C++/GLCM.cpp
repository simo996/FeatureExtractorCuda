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
    this->numberOfPairs = getBorderRows() * getBorderColumns();
    initializeGlcmElements();
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

int GLCM::getBorderRows() const{
   return (windowData.dimension - (windowData.distance * abs(windowData.shiftRows)));
}

int GLCM::getBorderColumns() const{
    return (windowData.dimension - (windowData.distance * abs(windowData.shiftColumns)));
}

void GLCM::printGLCMData() const{
    cout << endl;
    cout << "***\tGLCM Data\t***" << endl;
    cout << "Shift rows : " << windowData.shiftRows << endl;
    cout << "Shift columns: " << windowData.shiftColumns  << endl;
    cout << "Father Window dimension: "<< windowData.dimension  << endl;
    cout << "Border Rows: "<< getBorderRows()  << endl;
    cout << "Border Columns: " << getBorderColumns()  << endl;
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
    for(MapIterator mi=grayPairsMap.begin(); mi!=grayPairsMap.end(); mi++)
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
inline int GLCM::getReferenceIndex(const int windowStartOffset, const int i, const int j,
                                   const int initialWindowRowOffset, const int initialWindowColumnOffset){
    int index = (((i + windowStartOffset) + initialWindowRowOffset) * img.getRows())
            + ((j + windowStartOffset) + initialWindowColumnOffset);
    assert(index >= 0);
    return index;
}

// addressing method for neighbor pixel; see documentation
inline int GLCM::getNeighborIndex(const int windowStartOffset, const int i, const int j,
                                  const int initialWindowColumnOffset){
    int index = ((i + windowStartOffset) * img.getColumns()) +
            ((j + windowStartOffset) + initialWindowColumnOffset + windowData.shiftColumns);
    assert(index >= 0);
    return index;
}
/*
    This method puts inside the map of elements map<GrayPair, int> each
    frequency associated with each pair of grayLevels
*/
void GLCM::initializeGlcmElements() {
    // TODO refactor these names for more clarity
    // Define subBorders offset depending on orientation
    int initialWindowColumnOffset = computeWindowColumnOffset();
    int initialWindowRowOffset = computeWindowRowOffset();
    // Offset to locate the starting point of the window inside the sliding image
    int windowStartOffset = (img.getRows() * windowData.imageRowsOffset) + windowData.imageColumnsOffset;

    int referenceGrayLevel;
    int neighborGrayLevel;
    for (int i = 0; i < getBorderRows() ; i++)
    {
        for (int j = 0; j < getBorderColumns(); j++)
        {
            // Extract the two pixels in the pair
            int referenceIndex = getReferenceIndex(windowStartOffset, i, j,
                    initialWindowRowOffset, initialWindowColumnOffset);
            referenceGrayLevel = img.getPixels()[referenceIndex];
            int neighborIndex = getNeighborIndex(windowStartOffset, i, j,
                    initialWindowColumnOffset);
            neighborGrayLevel = img.getPixels()[neighborIndex];

            GrayPair actualPair(referenceGrayLevel, neighborGrayLevel);
            grayPairsMap[actualPair] += 1;
            if(windowData.symmetric) // Create the symmetric counterpart
            {
                GrayPair simmetricPair(neighborGrayLevel, referenceGrayLevel);
                grayPairsMap[simmetricPair]+=1;
            }
            
        }
    }

}

/*
    This method, given the map<GrayPair, int freq> will produce 
    map<int k, int freq> where k is the sum of both grayLevels of the GrayPair.
    This representation is used in computeSumXXX() features
*/
map<AggregatedGrayPair, int> GLCM::codifySummedPairs() const{
    map<AggregatedGrayPair, int> aggregatedPairs;
    
    typedef map<GrayPair, int>::const_iterator MapIterator;
    for(MapIterator mi=grayPairsMap.begin(); mi!=grayPairsMap.end(); mi++)
    {
        int k= mi->first.getGrayLevelI() + mi->first.getGrayLevelJ();
        AggregatedGrayPair element(k);

        aggregatedPairs[element]+=mi->second;
    }
    return aggregatedPairs;
}

/*
    This method, given the map<GrayPair, int freq> will produce 
    map<int k, int freq> where k is the absolute difference of both grayLevels
    of the GrayPair.
    This representation is used in computeDiffXXX() features
*/
map<AggregatedGrayPair, int> GLCM::codifySubtractedPairs() const{
    map<AggregatedGrayPair, int> aggregatedPairs;
    
    typedef map<GrayPair, int>::const_iterator MapIterator;
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

/*
    This method, given the map<GrayPair, int freq> will produce 
    map<int k, int freq> where k is the REFERENCE grayLevel of the GrayPair 
    while freq is the "marginal" frequency of that level 
    (ie. how many times k is present in all GrayPair<k, ?>)
    This representation is used for computing features HX, HXY, HXY1, imoc
*/
map<int, int> GLCM::codifyXMarginalProbabilities() const{
    map<int, int> xMarginalPairs;

    typedef map<GrayPair, int>::const_iterator GrayPairsMapIterator;
    for(GrayPairsMapIterator mi=grayPairsMap.begin(); mi!=grayPairsMap.end(); mi++)
    {
        int firstGrayPair = mi->first.getGrayLevelI();
        xMarginalPairs[firstGrayPair] += mi->second;
    }

    return xMarginalPairs;
}

/*
    This method, given the map<GrayPair, int freq> will produce 
    map<int k, int freq> where k is the NEIGHBOR grayLevel of the GrayPair 
    while freq is the "marginal" frequency of that level 
    (ie. how many times k is present in all GrayPair<?, k>)
    This representation is used for computing features HX, HXY, HXY1, imoc
*/
map<int, int> GLCM::codifyYMarginalProbabilities() const{
    map<int, int> yMarginalPairs;
    
    typedef map<GrayPair, int>::const_iterator GrayPairsMapIterator;
    for(GrayPairsMapIterator mi=grayPairsMap.begin(); mi!=grayPairsMap.end(); mi++)
    {
        int secondGrayPair= mi->first.getGrayLevelJ();
        yMarginalPairs[secondGrayPair]+=mi->second;
    }

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
