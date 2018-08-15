//
// Created by simo on 11/07/18.
//

#include <iostream>
#include <algorithm>
#include <assert.h>
#include "GLCM.h"
#include "GrayPair.h"
#include "AggregatedGrayPair.h"

using namespace std;

// Constructors
GLCM::GLCM(const unsigned int * pixels, const ImageData& image,
        Window& windowData, WorkArea wa): pixels(pixels), img(image),
        windowData(windowData), workArea(wa){
    this->numberOfPairs = getBorderRows() * getBorderColumns();
    if(this->windowData.symmetric)
        this->numberOfPairs *= 2;
    elements = vector<GrayPair> (numberOfPairs);
    summedPairs = vector<AggregatedGrayPair> (numberOfPairs);
    subtractedPairs = vector<AggregatedGrayPair> (numberOfPairs);
    xMarginalPairs = vector<AggregatedGrayPair> (numberOfPairs);
    yMarginalPairs = vector<AggregatedGrayPair> (numberOfPairs);

    initializeGlcmElements();
}

// Set the working area to initial condition
GLCM::~GLCM(){
    workArea.cleanup();
}

GrayPair * GLCM::getGrayPairs() {
    return workArea.getGrayPairs();
}

AggregatedGrayPair * GLCM::getSummedPairs(){
    return workArea.getSummedPairs();
}

AggregatedGrayPair * GLCM::getSubtractedPairs(){
    return workArea.getSubtractedPairs();
}

AggregatedGrayPair * GLCM::getxMarginalPairs(){
    return workArea.getxMarginalPairs();
}

AggregatedGrayPair * GLCM::getyMarginalPairs(){
    return workArea.getyMarginalPairs();
}

// Warning, se simmetrica lo spazio deve raddoppiare
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
   return (windowData.side - (windowData.distance * abs(windowData.shiftRows)));
}

int GLCM::getBorderColumns() const{
    return (windowData.side - (windowData.distance * abs(windowData.shiftColumns)));
}

void GLCM::printGLCMData() const{
    cout << endl;
    cout << "***\tGLCM Data\t***" << endl;
    cout << "Shift rows : " << windowData.shiftRows << endl;
    cout << "Shift columns: " << windowData.shiftColumns  << endl;
    cout << "Father Window side: "<< windowData.side  << endl;
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
    int lenght = getNumberOfUniquePairs();
    for (int i = 0; i < lenght; ++i) {
        elements[i].printPair();;
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

inline void insertElement(vector<GrayPair>& elements, const GrayPair actualPair, uint& lastInsertionPosition){
    auto position = find(elements.begin(), elements.end(), actualPair);
    // If found
    if((lastInsertionPosition > 0) // 0,0 as first element will increase insertion position
        && (position != elements.end())){ // if the item was already inserted
        position.operator*().operator++();
        if((actualPair.getGrayLevelI() == 0) && (actualPair.getGrayLevelJ() == 0)
            && (position.operator*().getFrequency() == actualPair.getFrequency()))
            // corner case, inserted pair 0,0 that matches with every empty field
            lastInsertionPosition++;
    }
    else
    {
        elements.at(lastInsertionPosition) = actualPair;
        lastInsertionPosition++;
    }
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
    unsigned int lastInsertionPosition = 0;
    for (int i = 0; i < getBorderRows() ; i++)
    {
        for (int j = 0; j < getBorderColumns(); j++)
        {
            // Extract the two pixels in the pair
            int referenceIndex = getReferenceIndex(i, j,
                    initialWindowRowOffset, initialWindowColumnOffset);
            referenceGrayLevel = pixels[referenceIndex];
            int neighborIndex = getNeighborIndex(i, j,
                    initialWindowColumnOffset);
            neighborGrayLevel = pixels[neighborIndex];

            GrayPair actualPair(referenceGrayLevel, neighborGrayLevel);
            insertElement(elements, actualPair, lastInsertionPosition);

            if(windowData.symmetric) // Create the symmetric counterpart
            {
                GrayPair simmetricPair(neighborGrayLevel, referenceGrayLevel);
                insertElement(elements, simmetricPair, lastInsertionPosition);
            }
            
        }
    }
    codifySummedPairs();
    codifySubtractedPairs();
    codifyXMarginalProbabilities();
    codifyYMarginalProbabilities();
}

unsigned int GLCM::getNumberOfUniquePairs() const{
    unsigned int i = 0;
    while((elements[i].getFrequency() !=0 ) && (i < numberOfPairs))
        i++;
    return i;
}

unsigned int GLCM::getNumberOfUniqueAggregatedElements(const vector<AggregatedGrayPair>& src) const{
    unsigned int i = 0;
    while((src[i].getFrequency() !=0 ) && (i < numberOfPairs))
        i++;
    return i;
}

inline void insertElement(vector<AggregatedGrayPair>& elements, const AggregatedGrayPair actualPair, uint& lastInsertionPosition){
    auto position = find(elements.begin(), elements.end(), actualPair);
    // If found
    if((lastInsertionPosition > 0) && // corner case 0 as first elment
        (position != elements.end())){ // if the item was already inserted
            position.operator*().increaseFrequency(actualPair.getFrequency());
        if((actualPair.getAggregatedGrayLevel() == 0) && // corner case 0 as regular element
        (position.operator*().getFrequency() == actualPair.getFrequency()))
            // corner case, inserted 0 that matches with every empty field
            lastInsertionPosition++;
    }
    else
    {
        elements.at(lastInsertionPosition) = actualPair;
        lastInsertionPosition++;
    }
}

/*
    This method, given the map<GrayPair, int freq> will produce 
    map<int k, int freq> where k is the sum of both grayLevels of the GrayPair.
    This representation is used in computeSumXXX() features
*/
void GLCM::codifySummedPairs() {
    unsigned int lastInsertPosition = 0;

    for(int i = 0 ; i < getNumberOfUniquePairs(); i++){
        uint k= elements[i].getGrayLevelI() + elements[i].getGrayLevelJ();
        AggregatedGrayPair element(k, elements[i].getFrequency());

        insertElement(summedPairs, element, lastInsertPosition);
    }
}

/*
    This method, given the map<GrayPair, int freq> will produce 
    map<int k, int freq> where k is the absolute difference of both grayLevels
    of the GrayPair.
    This representation is used in computeDiffXXX() features
*/
void GLCM::codifySubtractedPairs() {
    unsigned int lastInsertPosition = 0;

    for(int i = 0 ; i < getNumberOfUniquePairs(); i++){
        int diff = elements[i].getGrayLevelI() - elements[i].getGrayLevelJ();
        uint k= static_cast<uint>(abs(diff));
        AggregatedGrayPair element(k, elements[i].getFrequency());

        insertElement(subtractedPairs, element, lastInsertPosition);
    }
}

void GLCM::printAggregated() const{
    printGLCMAggregatedElements(true);
    printGLCMAggregatedElements(false);
}

void GLCM::printGLCMAggregatedElements(bool areSummed) const{
    cout << endl;
    if(areSummed) {
        cout << "* Summed grayPairsMap *" << endl;
        int length = getNumberOfUniqueAggregatedElements(summedPairs);
        for (int i = 0; i < length; ++i) {
            summedPairs[i].printPair();
        }
    }
    else {
        cout << "* Subtracted grayPairsMap *" << endl;
        int length = getNumberOfUniqueAggregatedElements(subtractedPairs);
        for (int i = 0; i < length; ++i) {
            subtractedPairs[i].printPair();
        }
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
    unsigned int lastInsertPosition = 0;

    for(int i = 0 ; i < getNumberOfUniquePairs(); i++){
        uint firstGrayLevel = elements[i].getGrayLevelI();
        AggregatedGrayPair element(firstGrayLevel, elements[i].getFrequency());

        insertElement(xMarginalPairs, element, lastInsertPosition);
    }
}

/*
    This method, given the map<GrayPair, int freq> will produce 
    map<int k, int freq> where k is the NEIGHBOR grayLevel of the GrayPair 
    while freq is the "marginal" frequency of that level 
    (ie. how many times k is present in all GrayPair<?, k>)
    This representation is used for computing features HX, HXY, HXY1, imoc
*/
void GLCM::codifyYMarginalProbabilities(){
    unsigned int lastInsertPosition = 0;

    for(int i = 0 ; i < getNumberOfUniquePairs(); i++){
        uint secondGrayLevel = elements[i].getGrayLevelJ();
        AggregatedGrayPair element(secondGrayLevel, elements[i].getFrequency());

        insertElement(yMarginalPairs, element, lastInsertPosition);
    }

}

void GLCM::printMarginalProbabilityElements() const{
    cout << endl << "* xMarginal Codifica" << endl;
    for (int i = 0; i < getNumberOfUniqueAggregatedElements(xMarginalPairs); ++i) {
        cout << "(" << xMarginalPairs[i].getAggregatedGrayLevel() <<
            ", X):\t" << xMarginalPairs[i].getFrequency() << endl;
    }
    cout << endl << "* yMarginal Codifica" << endl;
    for (int i = 0; i < getNumberOfUniqueAggregatedElements(yMarginalPairs); ++i) {
        cout << "(X, " << yMarginalPairs[i].getAggregatedGrayLevel() << ")" <<
            ":\t" << yMarginalPairs[i].getFrequency() << endl;

    }

}


