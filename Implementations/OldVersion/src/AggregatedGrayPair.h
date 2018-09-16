#ifndef FEATUREEXTRACTOR_AGGREGATEDGRAYPAIR_H
#define FEATUREEXTRACTOR_AGGREGATEDGRAYPAIR_H

/*
    This class represent two possible type of elements:
    - Elements obtained by summing or subtracting 2 gray levels of a pixel pair
    - Elements representing the frequency of 1 of the 2 gray levels of the
    pixel pairs (reference gray level or neighbor gray level)
*/

class AggregatedGrayPair {
public:
    explicit AggregatedGrayPair(unsigned int i);

    // show textual representation
    void printPair() const;

    // Getters
    int getAggregatedGrayLevel() const;

    bool operator<(const AggregatedGrayPair& other) const{
        return (grayLevel < other.getAggregatedGrayLevel());
    }
private:
    unsigned int grayLevel;

};


#endif //FEATUREEXTRACTOR_AGGREGATEDGRAYPAIR_H
