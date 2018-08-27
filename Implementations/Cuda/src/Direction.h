/*
 * Direction.h
 *
 *  Created on: 26/ago/2018
 *      Author: simone
 */

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

class Direction {
public:
    CUDA_HOSTDEV Direction(int directionNumber);
    CUDA_HOSTDEV static void printDirectionLabel(const int direction);
    char label[21];
    int shiftRows;
    int shiftColumns;
};
#endif /* DIRECTION_H_ */
