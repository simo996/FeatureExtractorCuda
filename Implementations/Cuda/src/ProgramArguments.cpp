/*
 * ProgramArguments.cpp
 *
 *  Created on: 26/ago/2018
 *      Author: simone
 */

#include "ProgramArguments.h"

#include <getopt.h> // For options check

void ProgramArguments::printProgramUsage(){
    cout << endl << "Usage: FeatureExtractor [<-s>] [<-i>] [<-d distance>] [<-w windowSize>] [<-n numberOfDirections>] "
                    "imagePath" << endl;
    exit(2);
}

ProgramArguments ProgramArguments::checkOptions(int argc, char* argv[]){
    ProgramArguments progArg;
    int opt;
    while((opt = getopt(argc, argv, "sw:d:in:hc")) != -1){
        switch (opt){
            case 'c':{
                // Crop original dynamic resolution
                progArg.crop = true;
                break;
            }
            case 's':{
                // Make the glcm pairs symmetric
                progArg.symmetric = true;
                break;
            }
            case 'i':{
                // Create images associated to features
                progArg.createImages = true;
                break;
            }
            case 'd': {
                // Choose the distance between pixels
                int distance = atoi(optarg);
                if (distance < 1) {
                    cout << "ERROR ! The distance between every pixel pair must be >= 1 ";
                    printProgramUsage();
                }
                progArg.distance = distance;
                break;
            }
            case 'w': {
                // Decide what the size of each sub-window of the image will be
                short int windowSize = atoi(optarg);
                if ((windowSize < 2) || (windowSize > 10000)) {
                    cout << "ERROR ! The size of the sub-windows to be extracted option (-w) "
                            "must have a value between 2 and 10000";
                    printProgramUsage();
                }
                progArg.windowSize = windowSize;
                break;
            }
            case 'n':{
                // Decide how many of the 4 directions will be copmuted
                short int dirNumber = atoi(optarg);
                if(dirNumber > 4 || dirNumber <1){
                    cout << "ERROR ! The type of directions to be computed "
                            "option (-n) must be a value between 1 and 4" << endl;
                    printProgramUsage();
                }
                progArg.directionType = dirNumber;
                break;
            }
            case '?':
                // Unrecognized options
                printProgramUsage();
            case 'h':
                // Help
                printProgramUsage();

                break;
            default:
                printProgramUsage();
        }

    }
    if(progArg.distance > progArg.windowSize){
        cout << "WARNING: distance can't be > of each window size; distance value corrected to 1" << endl;
        progArg.distance = 1;
    }
    // The last parameter must be the image path
    if(optind +1 == argc){
        cout << "imagepath: " << argv[optind];
        progArg.imagePath = argv[optind];
    } else{
        progArg.imagePath= "../../../SampleImages/brain1.tiff";
        /*
        cout << "Missing image path!" << endl;
        printProgramUsage();
        */
    }
    return progArg;
}
