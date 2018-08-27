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

class AggregatedGrayPair {
public:
    CUDA_DEV AggregatedGrayPair();
    CUDA_DEV AggregatedGrayPair(unsigned int i, unsigned int frequency);
    CUDA_DEV void printPair() const;
    CUDA_DEV int getAggregatedGrayLevel() const;
    CUDA_DEV unsigned int getFrequency() const;
    CUDA_DEV void increaseFrequency(unsigned int amount);
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
    unsigned int grayLevel;
    unsigned int frequency;

};

#endif /* AGGREGATEDGRAYPAIR_H_ */
