#include "ProgramArguments.h"

#include <getopt.h> // For options check

void ProgramArguments::printProgramUsage(){
    cout << endl << "Usage: FeatureExtractor [<-s>] [<-d distance>] [<-w windowSize>] [<-t directionType>] "
                    "[<-b borderType>] [<-g>][- i imagePath] [<-o outputFolder>]" << endl;
    exit(2);
}

ProgramArguments ProgramArguments::checkOptions(int argc, char* argv[]){
    ProgramArguments progArg;
    int opt;
    while((opt = getopt(argc, argv, "gsw:d:n:hct:vo:i:r:b:")) != -1){
        switch (opt){
            case 'b':{
                // Choose between no, zero or symmetric padding
                short int type = atoi(optarg);
                switch(type){
                    case 0:
                    case 1:
                    case 2:
                        progArg.borderType = type;
                        break;
                    default:
                        cerr << "ERROR! -b option must be a value between 0 and 2" << endl;
                        printProgramUsage();
                }
                break;
            }
            case 'r':{
                // Crop original dynamic resolution
                progArg.quantitize = true;
                progArg.quantitizationMax = atoi(optarg);
                break;
            }
            case 'g':{
                // Make the glcm pairs symmetric
                progArg.symmetric = true;
            }
            case 's':{
                // Create images associated to features
                progArg.createImages = true;
                break;
            }
            case 'i':{
                // Folder where to put results
                progArg.imagePath = optarg;
                break;
            }
            case 'o':{
                // Folder where to put results
                progArg.outputFolder = optarg;
                break;
            }
            case 'v':{
                // Verbosity
                progArg.verbose = true;
                break;
            }
            case 'd': {
                int distance = atoi(optarg);
                if (distance < 1) {
                    cerr << "ERROR ! The distance between every pixel pair must be >= 1 ";
                    printProgramUsage();
                }
                progArg.distance = distance;
                break;
            }
            case 'w': {
                // Decide what the size of each sub-window of the image will be
                short int windowSize = atoi(optarg);
                if ((windowSize < 2) || (windowSize > 10000)) {
                    cerr << "ERROR ! The size of the sub-windows to be extracted option (-w) "
                            "must have a value between 2 and 10000";
                    printProgramUsage();
                }
                progArg.windowSize = windowSize;
                break;
            }
            case 't':{
                // Decide how many of the 4 directions will be computed
                short int dirType = atoi(optarg);
                if(dirType > 4 || dirType <1){
                    cerr << "ERROR ! The type of directions to be computed "
                            "option (-t) must be a value between 1 and 4" << endl;
                    printProgramUsage();
                }
                progArg.directionType = dirType;

                break;
            }
            case 'n':{
                short int dirNumber = atoi(optarg);
                if(dirNumber != 1){
                    cout << "Warning! At this moment just 1 direction "
                            "can be be computed at each time" << endl;
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

    // No image provided
    if(progArg.imagePath.empty()) {
        cerr << "ERROR! Missing image path!" << endl;
        printProgramUsage();
    }

    // Option output folder was not used
    if(progArg.outputFolder.empty())
        progArg.outputFolder = Utils::removeExtension(Utils::basename(progArg.imagePath));

    return progArg;
}