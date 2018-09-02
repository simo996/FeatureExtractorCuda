//
// Created by simo on 11/07/18.
//

#ifndef FEATUREEXTRACTOR_AGGREGATEDGRAYPAIR_H
#define FEATUREEXTRACTOR_AGGREGATEDGRAYPAIR_H

// Custom types for easy future correction
// Unsigned shorts half the memory footprint of the application
typedef unsigned short grayLevelType;
typedef unsigned short frequencyType;

class AggregatedGrayPair {
public:
    AggregatedGrayPair();
    AggregatedGrayPair(grayLevelType i, frequencyType frequency);
    void printPair() const;
    grayLevelType getAggregatedGrayLevel() const;
    frequencyType getFrequency() const;
    bool compareTo(AggregatedGrayPair other) const;
    void increaseFrequency(frequencyType amount);

    bool operator==(const AggregatedGrayPair& other) const{
        return (grayLevel == other.getAggregatedGrayLevel());
    }


    bool operator<(const AggregatedGrayPair& other) const{
        return (grayLevel < other.getAggregatedGrayLevel());
    }

    AggregatedGrayPair& operator++(){
        this->frequency += 1;
        return *this;
    }
private:
    grayLevelType grayLevel;
    frequencyType frequency;

};


#endif //FEATUREEXTRACTOR_AGGREGATEDGRAYPAIR_H
