#ifndef FEATUREEXTRACTOR_WINDOW_H
#define FEATUREEXTRACTOR_WINDOW_H

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
    Window(short int dimension, short int distance, short int directionType, bool symmetric = false);
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
    void setDirectionShifts(int shiftRows, int shiftColumns);

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
    void setSpacialOffsets(int rowOffset, int columnOffset);
};


#endif //FEATUREEXTRACTOR_WINDOW_H
