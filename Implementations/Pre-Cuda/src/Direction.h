/*
	This class represent a supported direction for locating reference-neighbor 
	pixel pairs
*/

#ifndef FEATUREEXTRACTOR_DIRECTION_H
#define FEATUREEXTRACTOR_DIRECTION_H

using namespace std;

class Direction {
public:
    Direction(int directionNumber);
    static void printDirectionLabel(const int direction);
    char label[20];
    int shiftRows;
    int shiftColumns;
};


#endif //FEATUREEXTRACTOR_DIRECTION_H
