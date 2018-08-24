/*
	This class represent a supported direction for locating reference-neighbor 
	pixel pairs
*/

#ifndef FEATUREEXTRACTOR_DIRECTION_H
#define FEATUREEXTRACTOR_DIRECTION_H

#include <string>
#include <vector>

using namespace std;

class Direction {
public:
    Direction(string label, int shiftRows, int shiftColumns);
    static vector<Direction> getAllDirections();
    static string getDirectionLabel(const int direction);
    static void printDirectionLabel(int direction);
    const string label;
    const int shiftRows;
    const int shiftColumns;
    static void printDirectionLabel(const Direction& direction);

};

#endif //FEATUREEXTRACTOR_DIRECTION_H
