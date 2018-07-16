//
// Created by simo on 11/07/18.
//

#ifndef FEATUREEXTRACTOR_GRAYPAIR_H
#define FEATUREEXTRACTOR_GRAYPAIR_H


class GrayPair {
    public:
        void printPair() const;
        int getGrayLevelI() const;
        int getGrayLevelJ() const;
        GrayPair(int i, int j);

        bool operator<(const GrayPair& other) const{
            if(grayLevelI != other.getGrayLevelI())
                return (grayLevelI < other.getGrayLevelI());
            else
                return (grayLevelJ < other.getGrayLevelJ());
        }
    private:
    //  TODO versione alternativa con singolo int per entrambi
        int grayLevelI;
        int grayLevelJ;

};


#endif //FEATUREEXTRACTOR_GRAYPAIR_H
