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
	int dimension;
    int distance;
    bool symmetric;
    Window(int dimension, int distance, bool symmetric = false);
    // Directions offset to locate the pixel pair <reference,neighbor>
    // The 4 possible combinations are imposed after the creation of the window
    int shiftRows; 
    int shiftColumns;
    void setDirectionOffsets(int shiftRows, int shiftColumns);
     // Shift to locate the window inside the entire image
    int imageRowsOffset;
    int imageColumnsOffset;
};


#endif //FEATUREEXTRACTOR_WINDOW_H
