//
// Created by simo on 11/07/18.
//

#ifndef FEATUREEXTRACTOR_GRAYPAIR_H
#define FEATUREEXTRACTOR_GRAYPAIR_H


class GrayPair {
public:
    explicit GrayPair(unsigned int i, unsigned int j);
    unsigned int getGrayLevelI() const;
    unsigned int getGrayLevelJ() const;
    void printPair() const;

    bool operator<(const GrayPair& other) const{
        if(grayLevelI != other.getGrayLevelI())
            return (grayLevelI < other.getGrayLevelI());
        else
            return (grayLevelJ < other.getGrayLevelJ());
    }
private:
    //  TODO versione alternativa con singolo int per entrambi
        unsigned int grayLevelI;
        unsigned int grayLevelJ;

};


#endif //FEATUREEXTRACTOR_GRAYPAIR_H
