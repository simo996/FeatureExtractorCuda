#ifndef FEATUREEXTRACTOR_GRAYPAIR_H
#define FEATUREEXTRACTOR_GRAYPAIR_H

// Custom types for easy future correction
// Unsigned shorts half the memory footprint of the application
typedef unsigned short grayLevelType;
typedef unsigned short frequencyType;

/**
 * This class represent the gray levels of a pixel pair
*/
class GrayPair {
public:
    /**
     * Constructor for initializing pre-allocated work areas
     */
    GrayPair();
    /**
     * Constructor for effective gray-tone pairs
     * @param i grayLevel of the reference pixel of the pair
     * @param j grayLevel of the neighbor pixel of the pair
     */
    GrayPair(grayLevelType i, grayLevelType j);
    /**
     * Getter
     * @return the gray level of the reference pixel of the pair
     */
    grayLevelType getGrayLevelI() const;
    /**
     * Getter
     * @return the gray level of the neighbor pixel of the pair
     */
    grayLevelType getGrayLevelJ() const;
    /**
     * Getter
     * @return the frequency of the pair of gray levels in the glcm
     */
    frequencyType getFrequency() const;
    /**
     * method to determine equality based on the gray tones of the pair
     * @param other: graypair to compare
     * @return: true if both grayLevels of both items are the same
     */
    bool compareTo(GrayPair other) const;
    // Setter
    /**
     * DEPRECATED Setter. Use the ++ operator instad
     */
    void frequencyIncrease(); // frequency can be incremented only by 1
    /**
     * Show textual representation of the gray pair
     */
    void printPair() const;

    // C++ operators inherited from implementation that uses STL
    GrayPair& operator++(){
        this->frequency +=1;
        return *this;
    }

    bool operator==(const GrayPair& other) const{
        if((grayLevelI == other.getGrayLevelI()) &&
            (grayLevelJ == other.getGrayLevelJ()))
            return true;
        else
            return false;

    }

    bool operator<(const GrayPair& other) const{
        if(grayLevelI != other.getGrayLevelI())
            return (grayLevelI < other.getGrayLevelI());
        else
            return (grayLevelJ < other.getGrayLevelJ());
    }
private:
    grayLevelType grayLevelI;
    grayLevelType grayLevelJ;
    frequencyType frequency;

};


#endif //FEATUREEXTRACTOR_GRAYPAIR_H
