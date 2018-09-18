
#ifndef FEATUREEXTRACTOR_DIRECTION_H
#define FEATUREEXTRACTOR_DIRECTION_H

/**
 * This class represent a supported direction;
 * it embeds values for locating reference-neighbor pixel pairs
 * Supported directions with their number associated:
 * 0°[1], 45°[2], 90° [3], 135° [4]
*/

class Direction {
public:
    /**
     * Constructs the class putting into it the correct values
     * @param directionNumber: the number associated with the direction:
     * 0°[1], 45°[2], 90° [3], 135° [4]
     */
    Direction(int directionNumber);
    /**
     * Show info about the direction
     * @param direction: the number associated with the direction:
     * 0°[1], 45°[2], 90° [3], 135° [4]
     */
    static void printDirectionLabel(const int direction);
    char label[20];
    /**
     * shift on the y axis to locate the neighbor pixel
     */
    int shiftRows;
    /**
     * shift on the x axis to locate the neighbor pixel
     */
    int shiftColumns;
};


#endif //FEATUREEXTRACTOR_DIRECTION_H
