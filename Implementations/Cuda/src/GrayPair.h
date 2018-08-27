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

class GrayPair{
public:
	CUDA_DEV GrayPair();
	CUDA_DEV GrayPair(unsigned int i, unsigned int j);
	CUDA_DEV unsigned int getGrayLevelI() const;
	CUDA_DEV unsigned int getGrayLevelJ() const;
	CUDA_DEV unsigned int getFrequency() const;
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
        unsigned int grayLevelI;
        unsigned int grayLevelJ;
        unsigned int frequency;
};
#endif /* GRAYPAIR_H_ */

