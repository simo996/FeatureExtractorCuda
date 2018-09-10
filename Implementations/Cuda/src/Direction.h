#ifndef DIRECTION_H_
#define DIRECTION_H_

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__ 
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif

/*
    This class represent a supported direction;
    it embeds values for locating reference-neighbor pixel pairs

    Supported directions with their number associated:
    0°[1], 45°[2], 90° [3], 135° [4]
*/

class Direction {
public:
    CUDA_HOSTDEV Direction(int directionNumber);
    CUDA_HOSTDEV static void printDirectionLabel(const int direction);
    char label[21];
    // shift one the y axis to locate the neighbor pixel
    int shiftRows;
    // shift one the x axis to locate the neighbor pixel
    int shiftColumns;
};
#endif /* DIRECTION_H_ */
