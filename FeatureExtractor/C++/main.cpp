
#include <iostream>
#include <vector>
#include <assert.h>
#include <fstream>
#include <getopt.h> // For options check
#include "ImageFeatureComputer.h"

using namespace std;
using namespace cv;

Image useMockupMatrix(){
    // This methods won't work until a copy mechanism is implemented into Image
    // At distruction the pointer points to corrupted data
    int testData[] = {0,0,1,1,
                      1,0,1,1,
                      0,2,2,2,
                      2,2,3,3};
    // Load Image object
    int *image = testData;
    int rows = 4;
    int columns = 4;
    int maxGrayLevel = 4;
    Image img(image, rows, columns, maxGrayLevel);
    return img;
}

Mat readImage(const char* fileName){
    Mat inputImage;
    try{
        inputImage = imread(fileName, CV_LOAD_IMAGE_ANYDEPTH);
    }
    catch (cv::Exception& e) {
        const char *err_msg = e.what();
        cout << "Exception occurred: " << err_msg << endl;
    }

    return inputImage;
}


void showImage(const char* fileName){
    Mat inputImage = imread(fileName, IMREAD_GRAYSCALE);
    imshow("Output", inputImage);
    waitKey(0);
}

struct ProgramArguments{
    short int windowSize;
    bool symmetric;
    short int distance;
    short int numberOfDirections;
    bool createImages;
    short int chosenDevice; // 0 = gpu, 1=cpu, 'a'= auto
    string imagePath;

    ProgramArguments(short int windowSize = 4, bool symmetric = false,
            short int distance = 1, short int numberOfDirections = 4,
                    bool createImages = true, short int chosenDevice = 0)
            : windowSize(windowSize), symmetric(symmetric), distance(distance),
             numberOfDirections(numberOfDirections),
             createImages(createImages), chosenDevice(chosenDevice){}
};

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
        cout << "Missing image path!" << endl;
        printProgramUsage();
    }
    return progArg;
}


int main(int argc, char* argv[]) {
    cout << argv[0] << endl;
    //ProgramArguments pa=checkOptions(argc, argv);

    // Mockup Matrix
    int testData[] = {0,0,1,1,
                      1,0,1,1,
                      0,2,2,2,
                      2,2,3,3};
    // Load Image object
    int *image = testData;
    int rows = 4;
    int columns = 4;
    int maxGrayLevel = 4;
    //img.printElements();
    Image img(image, rows, columns, maxGrayLevel);

    /* NON VA UN CAZZO
    Mat save(rows, columns, CV_8UC1);
    memcpy(save.data, image, (rows*columns) * sizeof(char));
    cout << save << endl;
     */

    /* ANCHE PEGGIO
    Mat brain = readImage("../../SampleImages/30.tiff");
    cout << brain;
    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", brain );                   // Show our image inside it.
    waitKey(0);
    */

    // Load uniform window information (extracted from parameters
    int distance = 1;
    int windowDimension = 3;
    Window wData(windowDimension, distance);

    // Always check that
    int directionConsidered = 1;
    assert(directionConsidered <= 4 && directionConsidered >=0);

    // Start Creating the GLCMs
    ImageFeatureComputer ifc(img, wData);
    vector<WindowFeatures> fs= ifc.computeAllFeatures(directionConsidered);
    vector<map<FeatureNames, vector<double>>> formattedFeatures = ifc.getAllDirectionsAllFeatureValues(fs);

    //ifc.printAllDirectionsAllFeatureValues(formattedFeatures);
    //ifc.saveFeaturesToFiles(formattedFeatures);

    return 0;
}