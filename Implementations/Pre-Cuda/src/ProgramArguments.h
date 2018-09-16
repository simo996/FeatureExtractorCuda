#ifndef PRE_CUDA_PROGRAMARGUMENTS_H
#define PRE_CUDA_PROGRAMARGUMENTS_H

#include <string>
#include <iostream>
#include <getopt.h> // For options check

#include "Utils.h"

using namespace std;

/*
 * Class that gets and checks all the possible parameters to the problem
*/
class ProgramArguments {
public:
    // Side of each squared window that will be generated
    short int windowSize;
    // Eventual reduction of gray levels to range [0,Max]
    bool quantitize;
    int quantitizationMax;
    short int borderType;
    // Eventual symmetricity of the pairs of gray levels
    bool symmetric;
    // Modulus of the vector that links reference to neighbor pixel
    short int distance;
    // Which direction to compute between 0°, 45°, 90°, 135°
    short int directionType;
    // How many direction compute for each window. At the moment just 1
    short int directionsNumber;
    // Eventual generation of images from features values computed
    bool createImages;
    // Where to read the image
    string imagePath;
    // Where to put the results the image
    string outputFolder;
    // Print addition info
    bool verbose;

    // Constructor with default values
    ProgramArguments(short int windowSize = 4,
                     bool crop = false,
                     bool quantitize = false,
                     bool symmetric = false,
                     short int distance = 1,
                     short int dirType = 1,
                     short int dirNumber = 1,
                     bool createImages = false,
                     short int padding = 1,
                     bool verbose = false,
                     string outFolder = "")
            : windowSize(windowSize), borderType(padding), quantitize(quantitize), symmetric(symmetric), distance(distance),
              directionType(dirType), directionsNumber(dirNumber),
              createImages(createImages), outputFolder(outFolder),
              verbose(verbose){};
    static void printProgramUsage();
    static ProgramArguments checkOptions(int argc, char* argv[]);
};


#endif //PRE_CUDA_PROGRAMARGUMENTS_H