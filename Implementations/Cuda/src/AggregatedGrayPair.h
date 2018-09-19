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

/**
 * This class represent two possible type of elements:
 * - Elements obtained by summing or subtracting 2 gray levels of a pixel pair
 * - Elements representing the frequency of 1 of the 2 gray levels of the
 * pixel pairs (reference gray level or neighbor gray level)
*/

class AggregatedGrayPair {
public:
    /**
     * Constructor for initializing pre-allocated work areas
     */
    CUDA_DEV AggregatedGrayPair();
    /**
     * Constructor for effective gray-tone pairs
     * @param level: gray level of the object
     * @param frequency: frequency of the object
     */
    CUDA_DEV AggregatedGrayPair(grayLevelType grayLevel, frequencyType frequency);
    /**
     * show textual representation with level and frequency
     */
    CUDA_DEV void printPair() const; 
    /**
     * Getter
     * @return the grayLevel of the object
     */
    CUDA_DEV grayLevelType getAggregatedGrayLevel() const;
    /**
     * Getter
     * @return the frequency of the object
     */
    CUDA_DEV frequencyType getFrequency() const;
    /**
     * Setter
     * @param amount that will increment the frequency
     */
    CUDA_DEV void increaseFrequency(frequencyType amount);
    /**
     * Method to determine the equality with another object based on the
     * equality of gray levels
     * @param other: object of the same type
     * @return true if the 2 objects have the same gray level
     */
    CUDA_DEV bool compareTo(AggregatedGrayPair other) const;

    // C++ operators inherited from implementation that uses STL
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
