//
// Created by simo on 11/07/18.
//

#ifndef FEATUREEXTRACTOR_GRAYPAIR_H
#define FEATUREEXTRACTOR_GRAYPAIR_H

// Custom types for easy future correction
// Unsigned shorts half the memory footprint of the application
typedef unsigned short grayLevelType;
typedef unsigned short frequencyType;

class GrayPair {
public:
    GrayPair();
    GrayPair(grayLevelType i, grayLevelType j);
    grayLevelType getGrayLevelI() const;
    grayLevelType getGrayLevelJ() const;
    frequencyType getFrequency() const;
    bool compareTo(GrayPair other) const;
    void frequencyIncrease();
    void printPair() const;

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
