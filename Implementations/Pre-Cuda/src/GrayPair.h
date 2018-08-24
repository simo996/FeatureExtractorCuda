//
// Created by simo on 11/07/18.
//

#ifndef FEATUREEXTRACTOR_GRAYPAIR_H
#define FEATUREEXTRACTOR_GRAYPAIR_H


class GrayPair {
public:
    GrayPair();
    GrayPair(unsigned int i, unsigned int j);
    unsigned int getGrayLevelI() const;
    unsigned int getGrayLevelJ() const;
    unsigned int getFrequency() const;
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
        unsigned int grayLevelI;
        unsigned int grayLevelJ;
        unsigned int frequency;

};


#endif //FEATUREEXTRACTOR_GRAYPAIR_H
