/*
 * ProgramArguments.h
 *
 *  Created on: 26/ago/2018
 *      Author: simone
 */

#ifndef PROGRAMARGUMENTS_H_
#define PROGRAMARGUMENTS_H_

#include <string>
#include <iostream>

using namespace std;

class ProgramArguments {
public:
    short int windowSize;
    bool crop;
    bool symmetric;
    short int distance;
    short int numberOfDirections;
    bool createImages;
    string imagePath;

    ProgramArguments(short int windowSize = 4, bool crop = false, bool symmetric = false,
                     short int distance = 1, short int numberOfDirections = 4,
                     bool createImages = false)
            : windowSize(windowSize), crop(crop), symmetric(symmetric), distance(distance),
              numberOfDirections(numberOfDirections),
              createImages(createImages){};
    static void printProgramUsage();
    static ProgramArguments checkOptions(int argc, char* argv[]);


};

#endif /* PROGRAMARGUMENTS_H_ */