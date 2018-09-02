/*
 * GrayPair.h
 *
 *  Created on: 25/ago/2018
 *      Author: simone
 */

#ifndef GRAYPAIR_H_
#define GRAYPAIR_H_

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

class GrayPair{
public:
	CUDA_DEV GrayPair();
	CUDA_DEV GrayPair(grayLevelType i, grayLevelType j);
	CUDA_DEV grayLevelType getGrayLevelI() const;
	CUDA_DEV grayLevelType getGrayLevelJ() const;
	CUDA_DEV frequencyType getFrequency() const;
	CUDA_DEV void frequencyIncrease();
	CUDA_DEV bool compareTo(GrayPair other) const;
	CUDA_DEV void printPair() const;

	CUDA_DEV GrayPair& operator++(){
        this->frequency +=1;
        return *this;
    }

	CUDA_DEV bool operator==(const GrayPair& other) const{
        if((grayLevelI == other.getGrayLevelI()) &&
            (grayLevelJ == other.getGrayLevelJ()))
            return true;
        else
            return false;

    }

	CUDA_DEV bool operator<(const GrayPair& other) const{
        if(grayLevelI != other.getGrayLevelI())
            return (grayLevelI < other.getGrayLevelI());
        else
            return (grayLevelJ < other.getGrayLevelJ());
    }
private:
        grayLevelType grayLevelI;
        grayLevelType grayLevelJ;
        frequencyType frequency;
};
#endif /* GRAYPAIR_H_ */

