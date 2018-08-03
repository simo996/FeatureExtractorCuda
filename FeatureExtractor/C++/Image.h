/*
 * This class embeds pixels and image's metadata used by other components
*/

#ifndef FEATUREEXTRACTOR_IMAGE_H
#define FEATUREEXTRACTOR_IMAGE_H

class Image {
public:
    Image(const uint* pixels, uint rows, uint columns, uint mxGrayLevel)
            :pixels(pixels), rows(rows), columns(columns), maxGrayLevel(mxGrayLevel){};
    //Image(Mat& image);
    const uint * getPixels() const;
    uint getRows() const;
    uint getColumns() const;
    uint getMaxGrayLevel() const;
    void printElements() const;
private:
    const uint * pixels;
    const uint rows;
    const uint columns;
    const uint maxGrayLevel;
};


#endif //FEATUREEXTRACTOR_IMAGE_H
