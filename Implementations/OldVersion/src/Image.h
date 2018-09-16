/*
 * This class embeds pixels and image's metadata used by other components
*/

#ifndef FEATUREEXTRACTOR_IMAGE_H
#define FEATUREEXTRACTOR_IMAGE_H

#include <vector>

using namespace std;
class Image {
public:
    Image(vector<uint> pixels, uint rows, uint columns, uint mxGrayLevel)
            :pixels(pixels), rows(rows), columns(columns), maxGrayLevel(mxGrayLevel){};
    const vector<uint> getPixels() const;
    uint getRows() const;
    uint getColumns() const;
    uint getMaxGrayLevel() const;
    void printElements() const;
    // Should belong to private class
    vector<uint> pixels;
    const uint rows;
    const uint columns;
    const uint maxGrayLevel;
};


#endif //FEATUREEXTRACTOR_IMAGE_H
