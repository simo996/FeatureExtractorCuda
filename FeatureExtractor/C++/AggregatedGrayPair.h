//
// Created by simo on 11/07/18.
//

#ifndef FEATUREEXTRACTOR_AGGREGATEDGRAYPAIR_H
#define FEATUREEXTRACTOR_AGGREGATEDGRAYPAIR_H


class AggregatedGrayPair {
    public:
        void printPair() const;
        int getAggregatedGrayLevel() const;
        AggregatedGrayPair(int i);

        bool operator<(const AggregatedGrayPair& other) const{
            return (grayLevel < other.getAggregatedGrayLevel());
        }
    private:
        int grayLevel;

};


#endif //FEATUREEXTRACTOR_AGGREGATEDGRAYPAIR_H
