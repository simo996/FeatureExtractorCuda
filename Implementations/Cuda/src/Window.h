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

/**
 * This class embeds all the necessary metadata used from GLCM class to locate
 * the pixel pairs that need to be processed in this window, in 1 direction
*/
class Window {
public:
    /**
     * Initialization
     * @param dimension: side of each squared window
     * @param distance: modulus of vector reference-neighbor pixel pair
     * @param directionType: direction of interest to this window
     * @param symmetric: symmetricity of the graylevels of the window
     */
    CUDA_HOSTDEV Window(short int dimension, short int distance, short int directionType, bool symmetric = false);
    // Structural data uniform for all windows
    /**
     * side of each squared window
     */
    short int side; 
    /**
     * modulus of vector reference-neighbor pixel pair
     */
    short int distance;
    /**
     * eventual symmetricity of the pixel pair
     */
    bool symmetric;
    /**
     * This is a convenience attribute that WindowFeatureComputer uses
     * Redundant information obtainable from the combination of shiftRows and
     * shiftColumns
     */
    short int directionType;
    
    // Directions shifts to locate the pixel pair <reference,neighbor>
    // The 4 possible combinations are imposed after the creation of the window
    /**
     * shift on the y axis to locate the neighbor pixel
     */
    int shiftRows; 
    /**
     * shift on the x axis to locate the neighbor pixel
     */
    int shiftColumns;
    /**
     * Acquire both shifts on the x and y axis to locate the neighbor pixel of
     * the pair
     */
    CUDA_HOSTDEV void setDirectionShifts(int shiftRows, int shiftColumns);

     // Offset to locate the starting point of the window inside the entire image
    /**
     * First row of the image that belongs to this window
     */
    int imageRowsOffset;
    /**
     * First column of the image that belongs to this window
     */
    int imageColumnsOffset;
    /**
     * Acquire both shift that tells the window starting point in the image
     */
    CUDA_HOSTDEV void setSpacialOffsets(int rowOffset, int columnOffset);
};

#endif /* WINDOW_H_ */
