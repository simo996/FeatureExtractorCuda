/*
 * Window.h
 *
 *  Created on: 26/ago/2018
 *      Author: simone
 */

#ifndef WINDOW_H_
#define WINDOW_H_

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__ 
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif

class Window {
public:
	// Structural data uniform for all windows
	short int side; // side of each window
    short int distance; // modulus of vector reference-neighbor pixel pair
    bool symmetric;
    short int numberOfDirections;
    CUDA_HOSTDEV Window(short int dimension, short int distance, short int numberOfDirections, bool symmetric = false);
    // Directions shifts to locate the pixel pair <reference,neighbor>
    // The 4 possible combinations are imposed after the creation of the window
    int shiftRows;
    int shiftColumns;
    CUDA_HOSTDEV void setDirectionShifts(int shiftRows, int shiftColumns);
     // Offset to locate the starting point of the window inside the entire image
    int imageRowsOffset;
    int imageColumnsOffset;
    CUDA_HOSTDEV void setSpacialOffsets(int rowOffset, int columnOffset);
};

#endif /* WINDOW_H_ */
