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

void Direction::printDirectionLabel(const int direction){
    switch(direction){
        case 0:
            cout << endl << " * Direction 0° *" << endl;
            break;
        case 1:
            cout << endl << " * Direction 45° *" << endl;
            break;
        case 2:
            cout << endl << " * Direction 90° *" << endl;
            break;
        case 3:
            cout << endl << " * Direction 135° *" << endl;
            break;
        default:
            cerr << "Fatal Error! Unrecognized direction";
            exit(-1);
    }
}


void Direction::printDirectionLabel(const Direction& direction) {
    cout << endl << "* " << direction.label << " *" << endl;
}

