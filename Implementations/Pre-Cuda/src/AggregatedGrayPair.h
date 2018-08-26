//
// Created by simo on 11/07/18.
//

#ifndef FEATUREEXTRACTOR_AGGREGATEDGRAYPAIR_H
#define FEATUREEXTRACTOR_AGGREGATEDGRAYPAIR_H


class AggregatedGrayPair {
public:
    AggregatedGrayPair();
    AggregatedGrayPair(unsigned int i, unsigned int frequency);
    void printPair() const;
    int getAggregatedGrayLevel() const;
    unsigned int getFrequency() const;
    bool compareTo(AggregatedGrayPair other) const;
    void increaseFrequency(unsigned int amount);

    bool operator==(const AggregatedGrayPair& other) const{
        return (grayLevel == other.getAggregatedGrayLevel());
    }


    bool operator<(const AggregatedGrayPair& other) const{
        return (grayLevel < other.getAggregatedGrayLevel());
    }

    AggregatedGrayPair& operator++(){
        this->frequency +=1;
        return *this;
    }
private:
    unsigned int grayLevel;
    unsigned int frequency;

};


#endif //FEATUREEXTRACTOR_AGGREGATEDGRAYPAIR_H
