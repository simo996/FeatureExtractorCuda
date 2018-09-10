#ifndef PROGRAMARGUMENTS_H_
#define PROGRAMARGUMENTS_H_

#include <string>
#include <iostream>

using namespace std;

/* 
 * Class that gets and checks all the possible parameters to the problem
*/
class ProgramArguments {
public:
    // Side of each squared window that will be generated
    short int windowSize;
    // Eventual reduction of gray levels to range 0,255
    bool crop;
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

    // Constructor with default values
    ProgramArguments(short int windowSize = 4,
            bool crop = false,
            bool symmetric = false,
            short int distance = 1,
            short int dirType = 1,
            short int dirNumber = 1,
            bool createImages = false)
            : windowSize(windowSize), crop(crop), symmetric(symmetric), distance(distance),
              directionType(dirType), directionsNumber(dirNumber),
              createImages(createImages){};
    static void printProgramUsage();
    static ProgramArguments checkOptions(int argc, char* argv[]);
};
#endif /* PROGRAMARGUMENTS_H_ */
