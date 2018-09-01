/*
 * AggregatedGrayPair.h
 *
 *  Created on: 25/ago/2018
 *      Author: simone
 */

#ifndef AGGREGATEDGRAYPAIR_H_
#define AGGREGATEDGRAYPAIR_H_

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__ 
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif

// Custom types for easy future correction
// Unsigned shorts half the memory footprint of the application
typedef short unsigned grayLevelType;
typedef short unsigned frequencyType;

/*
    This class represent two possible type of elements:
    - Elements obtained by summing or subtracting 2 gray levels of a pixel pair
    - Elements representing the frequency of 1 of the 2 gray levels of the 
    pixel pairs (reference gray level or neighbor gray level)
*/ 

class AggregatedGrayPair {
public:
    CUDA_DEV AggregatedGrayPair();
    CUDA_DEV AggregatedGrayPair(grayLevelType grayLevel, frequencyType frequency);
    // show a representation
    CUDA_DEV void printPair() const;
    CUDA_DEV grayLevelType getAggregatedGrayLevel() const;
    CUDA_DEV frequencyType getFrequency() const;
    CUDA_DEV void increaseFrequency(frequencyType amount);
    CUDA_DEV bool compareTo(AggregatedGrayPair other) const;

    CUDA_DEV bool operator==(const AggregatedGrayPair& other) const{
        return (grayLevel == other.getAggregatedGrayLevel());
    }


    CUDA_DEV bool operator<(const AggregatedGrayPair& other) const{
        return (grayLevel < other.getAggregatedGrayLevel());
    }

    CUDA_DEV AggregatedGrayPair& operator++(){
        this->frequency +=1;
        return *this;
    }
private:
    grayLevelType grayLevel;
    frequencyType frequency;

};

#endif /* AGGREGATEDGRAYPAIR_H_ */
