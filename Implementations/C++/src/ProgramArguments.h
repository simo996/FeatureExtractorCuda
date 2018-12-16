#ifndef PRE_CUDA_PROGRAMARGUMENTS_H
#define PRE_CUDA_PROGRAMARGUMENTS_H

#include <string>
#include <iostream>
#include <getopt.h> // For options check

#include "Utils.h"

using namespace std;

/**
 * Class that gets, embeds and checks all the possible parameters to the problem
 */
class ProgramArguments {
public:
    /**
     * Side of each squared window that will be generated
     */
    short int windowSize;
    /**
     *  Optional reduction of gray levels to range [0,Max]
     */
    bool quantitize;
    /**
     *  Maximum gray level when a reduction of gray levels to range [0,Max]
     *  is applied
     */
    int quantitizationMax;
    /**
     * Type of border applied to the orginal image:
     * 0 = no border
     * 1 = zero pixel border
     * 2 = symmetric border
     */
    short int borderType;
    /**
     * Optional NON symmetricity of the pairs of gray levels in each glcm
     */
    bool symmetric;
    /**
     * Modulus of the vector that links reference to neighbor pixel
     */
    short int distance;
    /**
     * Which direction to compute between 0°, 45°, 90°, 135°
     */
    short int directionType;
    /**
     * How many direction compute for each window. LIMITED to 1 at this release
     */
    short int directionsNumber;
    /**
     * Optional generation of images from features values computed
     */
    bool createImages;
    /**
     * Path/name of the image to process; MANDATORY
     */
    string imagePath;
    /**
     * Where to put the results the image.
     * If none is provided the name of the image will be used without
     * path/extensions
     */
    string outputFolder;
    /**
     * Print additional information
     */
    bool verbose;

    /**
     * Constructor of the class that embeds all the parameters of the problem
     * @param windowSize: side of each squared window that will be created
     * @param quantitize: optional reduction of gray levels in [0, M]
     * @param symmetric: optional symmetricity of gray levels in each pixels pair
     * @param distance: modulus of the vector that links reference to neighbor pixel
     * @param dirType: Which direction to compute between 0°, 45°, 90°, 135°
     * @param dirNumber: how many direction will be computed simultaneously
     * @param createImages: optional generation of images from features values
     * computed
     * @param border: type of border applied to the orginal image
     * @param verbose: print additional info
     * @param outFolder: where to put results
     */
    ProgramArguments(short int windowSize = 4,
                     bool quantitize = false,
                     bool symmetric = true,
                     short int distance = 1,
                     short int dirType = 1,
                     short int dirNumber = 1,
                     bool createImages = false,
                     short int border = 1,
                     bool verbose = false,
                     string outFolder = "")
            : windowSize(windowSize), borderType(border), quantitize(quantitize), symmetric(symmetric), distance(distance),
              directionType(dirType), directionsNumber(dirNumber),
              createImages(createImages), outputFolder(outFolder),
              verbose(verbose){};
    /**
     * Show the user how to use the program and its options
     */
    static void printProgramUsage();
    /**
     * Loads the options given in the command line and checks them
     * @param argc
     * @param argv
     * @return
     */
    static ProgramArguments checkOptions(int argc, char* argv[]);
};


#endif //PRE_CUDA_PROGRAMARGUMENTS_H