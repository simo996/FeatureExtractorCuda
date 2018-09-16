#ifndef FEATUREEXTRACTOR_GRAYPAIR_H
#define FEATUREEXTRACTOR_GRAYPAIR_H

// Custom types for easy future correction
// Unsigned shorts half the memory footprint of the application
typedef unsigned short grayLevelType;
typedef unsigned short frequencyType;

/*
    This class represent the gray levels of a pixel pair
*/
class GrayPair {
public:
    // Constructor for initializing pre-allocated work areas
    GrayPair();
    // Constructor for effective gray-tone pairs
    GrayPair(grayLevelType i, grayLevelType j);
    // Getters
    grayLevelType getGrayLevelI() const;
    grayLevelType getGrayLevelJ() const;
    frequencyType getFrequency() const;
    // method to determine equality based on the gray tones of the pair
    bool compareTo(GrayPair other) const;
    // Setter
    void frequencyIncrease(); // frequency can be incremented only by 1
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
