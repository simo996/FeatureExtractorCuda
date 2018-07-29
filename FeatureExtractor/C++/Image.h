/*
 * La classe include i pixels e le informazioni di contorno dell'immagine
    Created by simo on 29/07/18.
*/

#ifndef FEATUREEXTRACTOR_IMAGE_H
#define FEATUREEXTRACTOR_IMAGE_H

class Image {
public:
    Image(const int* pixels, int rows, int columns, int mxGrayLevel)
            :pixels(pixels), rows(rows), columns(columns), maxGrayLevel(mxGrayLevel){};
    //Image(Mat& image);
    const int* getPixels() const;
    int getRows() const;
    int getColumns() const;
    int getMaxGrayLevel() const;
    void printElements() const;
private:
    const int * pixels;
    const int rows;
    const int columns;
    const int maxGrayLevel;
};


#endif //FEATUREEXTRACTOR_IMAGE_H
