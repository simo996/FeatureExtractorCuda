/*
	Responsabilità classe: contenere tutti i dati utili alla classe GLCM per 
	individuare i propri pixelPair
	Created by simo on 26/07/18.
*/

#ifndef FEATUREEXTRACTOR_WINDOW_H
#define FEATUREEXTRACTOR_WINDOW_H


class Window {
public:
	// Structural data uniform for all windows
	short int side; // side of each window
    short int distance; // modulus of vector reference-neighbor pixel pair 
    bool symmetric;
    Window(short int dimension, short int distance, bool symmetric = false);
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
