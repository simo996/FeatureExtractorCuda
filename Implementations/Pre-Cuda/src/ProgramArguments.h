//
// Created by simone on 26/08/18.
//

#ifndef PRE_CUDA_PROGRAMARGUMENTS_H
#define PRE_CUDA_PROGRAMARGUMENTS_H

#include <string>
#include <iostream>

using namespace std;

class ProgramArguments {
public:
    short int windowSize;
    bool crop;
    bool symmetric;
    short int distance;
    short int directionType;
    short int directionsNumber;
    bool createImages;
    string imagePath;

    ProgramArguments(short int windowSize = 4, bool crop = false, bool symmetric = false,
                     short int distance = 1, short int dirType = 1, short int dirNumber = 1,
                     bool createImages = false)
            : windowSize(windowSize), crop(crop), symmetric(symmetric), distance(distance),
              directionType(dirType), directionsNumber(dirNumber),
              createImages(createImages){};
    static void printProgramUsage();
    static ProgramArguments checkOptions(int argc, char* argv[]);

};


#endif //PRE_CUDA_PROGRAMARGUMENTS_H
