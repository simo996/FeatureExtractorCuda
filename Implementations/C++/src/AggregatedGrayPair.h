#ifndef FEATUREEXTRACTOR_AGGREGATEDGRAYPAIR_H
#define FEATUREEXTRACTOR_AGGREGATEDGRAYPAIR_H

// Custom types for easy future correction
// Unsigned shorts half the memory footprint of the application
typedef unsigned short grayLevelType;
typedef unsigned short frequencyType;

/*
    This class represent two possible type of elements:
    - Elements obtained by summing or subtracting 2 gray levels of a pixel pair
    - Elements representing the frequency of 1 of the 2 gray levels of the
    pixel pairs (reference gray level or neighbor gray level)
*/

class AggregatedGrayPair {
public:
    // Constructor for initializing pre-allocated work areas
    AggregatedGrayPair();
    // Constructor for effective gray-tone pairs
    AggregatedGrayPair(grayLevelType i, frequencyType frequency);
    // show textual representation
    void printPair() const;
    // Getters
    grayLevelType getAggregatedGrayLevel() const;
    frequencyType getFrequency() const;
    // Setter
    void increaseFrequency(frequencyType amount);
    // method to determine equality based on the gray tone
    bool compareTo(AggregatedGrayPair other) const;

    // C++ operators inherited from implementation that uses STL
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
