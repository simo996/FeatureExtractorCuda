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

/*
    This class embeds all the necessary metadata used from GLCM class to locate
    the pixel pairs that need to be processed
*/

class Window {
public:
    CUDA_HOSTDEV Window(short int dimension, short int distance, short int directionType, bool symmetric = false);
    // Structural data uniform for all windows
    short int side; // side of each window, that are squared
    short int distance; // modulus of vector reference-neighbor pixel pair
    bool symmetric; // eventual symmetricity of the pixel pair

    // This is a convenience attribute that WindowFeatureComputer uses
    short int directionType; // redundant information

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
