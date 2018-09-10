#ifndef FEATUREEXTRACTOR_WINDOW_H
#define FEATUREEXTRACTOR_WINDOW_H

/*
	This class embeds all the necessary metadata used from GLCM class to locate
    the pixel pairs that need to be processed
*/

class Window {
public:
    Window(short int dimension, short int distance, short int directionType, bool symmetric = false);
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
    void setDirectionShifts(int shiftRows, int shiftColumns);

     // Offset to locate the starting point of the window inside the entire image
    int imageRowsOffset;
    int imageColumnsOffset;
    void setSpacialOffsets(int rowOffset, int columnOffset);
};


#endif //FEATUREEXTRACTOR_WINDOW_H
