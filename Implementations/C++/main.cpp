
#include <iostream>
#include <vector>
#include <assert.h>
#include <fstream>
#include <getopt.h> // For options check
#include <opencv2/imgproc.hpp>
#include "ImageFeatureComputer.h"

using namespace std;

void printProgramUsage(){
    cout << endl << "Usage: FeatureExtractor <-s> <-i> <-d distance> <-w windowSize> <-n numberOfDirections> "
                    "imagePath" << endl;
    exit(2);
}

ProgramArguments checkOptions(int argc, char* argv[])
{
    ProgramArguments progArg;
    int opt;
    while((opt = getopt(argc, argv, "sw:d:in:h")) != -1){
        switch (opt){
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
                // Choose the distance between
                short int windowSize = atoi(optarg);
                if ((windowSize < 3) || (windowSize > 100)) {
                    printProgramUsage();
                }
                progArg.windowSize = windowSize;
                break;
            }
            case 'w': {
                // Decide what the size of each sub-window of the image will be
                short int windowSize = atoi(optarg);
                if ((windowSize < 3) || (windowSize > 100)) {
                    cout << "ERROR ! The size of the sub-windows to be extracted option (-w) "
                            "must have a value between 4 and 100";
                    printProgramUsage();
                }
                progArg.windowSize = windowSize;
                break;
            }
            case 'n':{
                // Decide how many of the 4 directions will be copmuted
                short int dirNumber = atoi(optarg);
                if(dirNumber > 4 || dirNumber <1){
                    cout << "ERROR ! The number of directions to be computed "
                            "option (-n) must be a value between 1 and 4" << endl;
                    printProgramUsage();
                }
                progArg.numberOfDirections = dirNumber;
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


int main(int argc, char* argv[]) {
    cout << argv[0] << endl;
    ProgramArguments pa = checkOptions(argc, argv);

    /*
    Mat brain = ImageLoader::readMriImage("../../../SampleImages/brain1.tiff");
    ImageLoader::printMatImageData(brain);
    ImageLoader::showImageStretched(brain, "brain1");
    */
    // MATLAB vedere se l'entropia viene = ad entropyfilt


    // Launch the external component
    ImageFeatureComputer ifc(pa);
    ifc.compute();


    return 0;
}