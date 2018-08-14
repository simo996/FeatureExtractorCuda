//
// Created by simo on 01/08/18.
//

#include <iostream>
#include "Direction.h"


Direction::Direction(string label, int shiftrws, int shiftcols)
        :label(label),shiftRows(shiftrws), shiftColumns(shiftcols){
}

vector<Direction> Direction::getAllDirections(){
    Direction d0{"Direction 0°", 0, 1};
    Direction d45{"Direction 45°", -1, 1};
    Direction d90{"Direction 90°", -1, 0};
    Direction d135{"Direction 135°", -1, -1};

    vector<Direction> out = {d0, d45, d90, d135};
    return out;
}

string Direction::getDirectionLabel(const int direction){
    switch(direction){
        case 0:
            return " * Direction 0° *" ;
        case 1:
            return " * Direction 45° *";
        case 2:
            return " * Direction 90° *";
        case 3:
            return " * Direction 135° *d";
        default:
            cerr << "Fatal Error! Unrecognized direction";
            exit(-1);
    }
}

void Direction::printDirectionLabel(const int direction)
    {
        cout << endl << getDirectionLabel(direction) << endl;
    }

void Direction::printDirectionLabel(const Direction& direction) {
    cout << endl << "* " << direction.label << " *" << endl;
}

