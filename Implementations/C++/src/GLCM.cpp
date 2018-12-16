#include <iostream>
#include <assert.h>
#include "GLCM.h"
#include "GrayPair.h"
#include "AggregatedGrayPair.h"

using namespace std;

// Constructors
GLCM::GLCM(const unsigned int * pixels, const ImageData& image,
        Window& windowData, WorkArea& wa): pixels(pixels), image(image),
        windowData(windowData),  workArea(wa) ,grayPairs(wa.grayPairs),
        summedPairs(wa.summedPairs), subtractedPairs(wa.subtractedPairs),
        xMarginalPairs(wa.xMarginalPairs), yMarginalPairs(wa.yMarginalPairs)
        {
    // Compute the number of pairs that need to be processed in this GLCM
    this->numberOfPairs = getWindowRowsBorder() * getWindowColsBorder();
    if(this->windowData.symmetric)
        this->numberOfPairs *= 2;

    // Replacing dirty memory with items that represent "available memory"
    workArea.cleanup();
    // Generate elements of this GLCM
    initializeGlcmElements();
}


// Set the working area to initial condition
GLCM::~GLCM(){

}

int GLCM::getNumberOfPairs() const {
        return numberOfPairs;
}

int GLCM::getMaxGrayLevel() const {
    return image.getMaxGrayLevel();
}

/**
 * Geometric limit of the sub-window
 * @return how many rows of the window need to be considered
 */
int GLCM::getWindowRowsBorder() const{
   return (windowData.side - (windowData.distance * abs(windowData.shiftRows)));
}

/**
 * Geometric limit of the sub-window
 * @return how many columns of the window need to be considered
 */
int GLCM::getWindowColsBorder() const{
    return (windowData.side - (windowData.distance * abs(windowData.shiftColumns)));
}


/**
 * Compute the shift to apply at the column for locating the pixels of each
 * pair of the glcm; it affects only 135° orientation
 * @return d (distance) pixels need to be ignored
 */
inline int GLCM::computeWindowColumnOffset()
{
    int initialColumnOffset = 0; // for 0°,45°,90°
    if((windowData.shiftRows * windowData.shiftColumns) > 0) // 135°
        initialColumnOffset = 1;
    return initialColumnOffset;
}

/**
 * Compute the shift to apply at the row for locating the pixels of each
 * pair of the glcm; it doesn't affect only 0° orientation
 * @return d (distance) pixels need to be ignored
*/
inline int GLCM::computeWindowRowOffset()
{
    int initialRowOffset = 1; // for 45°,90°,135°
    if((windowData.shiftRows == 0) && (windowData.shiftColumns > 0))
        initialRowOffset = 0; // for 0°
    return initialRowOffset;
}

/**
 * Addressing methods to get the reference pixel in each pair of the glcm
 * @param row in the sub-window of the reference pixel
 * @param col in the sub-window of the reference pixel
 * @param initialRowOffset see computeWindowRowOffset
 * @param initialColumnOffset see computeWindowColOffset
 * @return the index of the pixel in the array of pixels (linearized) of
 * the window
 */
inline int GLCM::getReferenceIndex(const int i, const int j,
                                   const int initialWindowRowOffset, const int initialWindowColumnOffset){
    int row = (i + windowData.imageRowsOffset) // starting point in the image
              + (initialWindowRowOffset * windowData.distance); // add direction shift
    int col = (j + windowData.imageColumnsOffset) + // starting point in the image
              (initialWindowColumnOffset * windowData.distance); // add direction shift
    int index = ( row * image.getColumns()) + col;
    assert(index >= 0);
    return index;
}

/**
 * Addressing methods to get the neighbor pixel in each pair of the glcm
 * @param row in the sub-window of the neighbor pixel
 * @param col in the sub-window of the neighbor pixel
 * @param initialColumnOffset see computeWindowColOffset
 * @return the index of the pixel in the array of pixels (linearized) of
 * the window
 */
inline int GLCM::getNeighborIndex(const int i, const int j,
                                  const int initialWindowColumnOffset){
    int row = (i + windowData.imageRowsOffset); // starting point in the image
    int col = (j + windowData.imageColumnsOffset) + // starting point in the image
              (initialWindowColumnOffset * windowData.distance) +  // add 135* right-shift
              (windowData.shiftColumns * windowData.distance); // add direction shift
    int index = (row * image.getColumns()) + col;
    assert(index >= 0);
    return index;
}

/**
 * Method that inserts a GrayPair in the pre-allocated memory
 * Uses that convention that GrayPair ( i=0, j=0, frequency=0) means
 * available memory
 */
inline void GLCM::insertElement(GrayPair* elements, const GrayPair actualPair,
        uint& lastInsertionPosition, bool symmetricity){
    int position = 0;
    // Find if the element was already inserted, and where
    while((!elements[position].compareTo(actualPair, symmetricity)) && (position < numberOfPairs))
        position++;
    // If found
    if((lastInsertionPosition > 0) // 0,0 as first element will increase insertion position
        && (position != numberOfPairs)){ // if the item was already inserted
        elements[position].operator++();
        if((actualPair.getGrayLevelI() == 0) && (actualPair.getGrayLevelJ() == 0)
            && (elements[position].getFrequency() == actualPair.getFrequency()))
            // corner case, inserted pair 0,0 that matches with every empty field
            lastInsertionPosition++;
    }
    else
    {
        elements[lastInsertionPosition] = actualPair;
        lastInsertionPosition++;
    }
}

/**
 * This method creates array of GrayPairs
*/
void GLCM::initializeGlcmElements() {
    // Define subBorders offset depending on orientation
    int initialWindowColumnOffset = computeWindowColumnOffset();
    int initialWindowRowOffset = computeWindowRowOffset();

    grayLevelType referenceGrayLevel;
    grayLevelType neighborGrayLevel;
    unsigned int lastInsertionPosition = 0;
    // Navigate the sub-window of interest
    for (int i = 0; i < getWindowRowsBorder() ; i++)
    {
        for (int j = 0; j < getWindowColsBorder(); j++)
        {
            // Extract the two pixels in the pair
            int referenceIndex = getReferenceIndex(i, j,
                    initialWindowRowOffset, initialWindowColumnOffset);
            // Application limit: only up to 2^16 gray levels
            referenceGrayLevel = pixels[referenceIndex]; // should be safe
            int neighborIndex = getNeighborIndex(i, j,
                    initialWindowColumnOffset);
            // Application limit: only up to 2^16 gray levels
            neighborGrayLevel = pixels[neighborIndex];  // should be safe

            GrayPair actualPair;
            if((windowData.symmetric) && (referenceGrayLevel > neighborGrayLevel)){
                actualPair = GrayPair(neighborGrayLevel, referenceGrayLevel);
            }
            else{
                actualPair = GrayPair(referenceGrayLevel, neighborGrayLevel);
            }
            insertElement(grayPairs, actualPair, lastInsertionPosition, windowData.symmetric);

        }
    }
    effectiveNumberOfGrayPairs = lastInsertionPosition;
    codifyAggregatedPairs();
    codifyMarginalPairs();
}

/**
 * Method that inserts a AggregatedGrayPair in the pre-allocated memory
 * Uses that convention that AggregateGrayPair (k=0, frequency=0) means
 * available memory
 */
inline void GLCM::insertElement(AggregatedGrayPair* elements, const AggregatedGrayPair actualPair, uint& lastInsertionPosition){
    int position = 0;
    // Find if the element was already inserted, and where
    while((!elements[position].compareTo(actualPair)) && (position < numberOfPairs))
        position++;
    // If found
    if((lastInsertionPosition > 0) && // corner case 0 as first elment
        (position != numberOfPairs)){ // if the item was already inserted
            elements[position].increaseFrequency(actualPair.getFrequency());
        if((actualPair.getAggregatedGrayLevel() == 0) && // corner case 0 as regular element
        (elements[position].getFrequency() == actualPair.getFrequency()))
            // corner case, inserted 0 that matches with every empty field
            lastInsertionPosition++;
    }
    else
    {
        elements[lastInsertionPosition] = actualPair;
        lastInsertionPosition++;
    }
}

/**
 * This method will produce the 2 arrays of AggregatedPairs (k, frequency)
 * where k is the sum or difference of both grayLevels of 1 GrayPair.
 * This representation is used in computeSumXXX() and computeDiffXXX() features
*/
void GLCM::codifyAggregatedPairs() {
    unsigned int lastInsertPosition = 0;
    // summed pairs first
    for(int i = 0 ; i < effectiveNumberOfGrayPairs; i++){
        // Create summed pairs first
        grayLevelType k= grayPairs[i].getGrayLevelI() + grayPairs[i].getGrayLevelJ();
        AggregatedGrayPair summedElement(k, grayPairs[i].getFrequency());

        insertElement(summedPairs, summedElement, lastInsertPosition);
    }
    numberOfSummedPairs = lastInsertPosition;

    // diff pairs
    lastInsertPosition = 0;
    for(int i = 0 ; i < effectiveNumberOfGrayPairs; i++){
        int diff = grayPairs[i].getGrayLevelI() - grayPairs[i].getGrayLevelJ();
        grayLevelType k= static_cast<uint>(abs(diff));
        AggregatedGrayPair element(k, grayPairs[i].getFrequency());

        insertElement(subtractedPairs, element, lastInsertPosition);
    }
    numberOfSubtractedPairs = lastInsertPosition;
}


/**
 * This method will produce the 2 arrays of AggregatedPairs (k, frequency)
 * where k is one grayLevel of GLCM and frequency is the "marginal" frequency of that level
 * (ie. how many times k is present in all GrayPair<k, ?>)
 * This representation is used for computing features HX, HXY, HXY1, imoc
*/
void GLCM::codifyMarginalPairs() {
    unsigned int lastInsertPosition = 0;
    // xMarginalPairs first
    // X Marginal pairs consider the reference gray level of the pixel pairs
    for(int i = 0 ; i < effectiveNumberOfGrayPairs; i++){
        grayLevelType firstGrayLevel = grayPairs[i].getGrayLevelI();
        AggregatedGrayPair element(firstGrayLevel, grayPairs[i].getFrequency());

        insertElement(xMarginalPairs, element, lastInsertPosition);
    }
    numberOfxMarginalPairs = lastInsertPosition;

    // yMarginalPairs second
    // Y Marginal pairs consider the neighbor gray level of the pixel pairs
    lastInsertPosition = 0;
    for(int i = 0 ; i < effectiveNumberOfGrayPairs; i++){
        grayLevelType secondGrayLevel = grayPairs[i].getGrayLevelJ();
        AggregatedGrayPair element(secondGrayLevel, grayPairs[i].getFrequency());

        insertElement(yMarginalPairs, element, lastInsertPosition);
    }
    numberOfyMarginalPairs = lastInsertPosition;
}


/* DEBUGGING METHODS */
void GLCM::printGLCM() const {
    printGLCMData();
    printGLCMElements();
    printAggregated();
    printMarginalProbabilityElements();
}

void GLCM::printGLCMData() const{
    cout << endl;
    cout << "***\tGLCM Data\t***" << endl;
    cout << "Shift rows : " << windowData.shiftRows << endl;
    cout << "Shift columns: " << windowData.shiftColumns  << endl;
    cout << "Father Window side: "<< windowData.side  << endl;
    cout << "Border Rows: "<< getWindowRowsBorder()  << endl;
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
    for (int i = 0; i < effectiveNumberOfGrayPairs; ++i) {
        grayPairs[i].printPair();;
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
        for (int i = 0; i < numberOfSummedPairs; ++i) {
            summedPairs[i].printPair();
        }
    }
    else {
        cout << "* Subtracted grayPairsMap *" << endl;
        for (int i = 0; i < numberOfSubtractedPairs; ++i) {
            subtractedPairs[i].printPair();
        }
    }
}

void GLCM::printMarginalProbabilityElements() const{
    cout << endl << "* xMarginal Codifica" << endl;
    for (int i = 0; i < numberOfxMarginalPairs; ++i) {
        cout << "(" << xMarginalPairs[i].getAggregatedGrayLevel() <<
            ", X):\t" << xMarginalPairs[i].getFrequency() << endl;
    }
    cout << endl << "* yMarginal Codifica" << endl;
    for (int i = 0; i <numberOfyMarginalPairs; ++i) {
        cout << "(X, " << yMarginalPairs[i].getAggregatedGrayLevel() << ")" <<
            ":\t" << yMarginalPairs[i].getFrequency() << endl;

    }

}


