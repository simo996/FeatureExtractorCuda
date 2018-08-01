//
// Created by simo on 01/08/18.
//

#ifndef FEATUREEXTRACTOR_DIRECTION_H
#define FEATUREEXTRACTOR_DIRECTION_H

#include <string>
#include <vector>

using namespace std;

class Direction {
public:
    const string label;
    const int shiftRows;
    const int shiftColumns;
    Direction(string label, int shiftRows, int shiftColumns);
    static vector<Direction> getAllDirections();
    static void printDirectionLabel(const Direction& direction);
    static void printDirectionLabel(int direction);

};


#endif //FEATUREEXTRACTOR_DIRECTION_H
