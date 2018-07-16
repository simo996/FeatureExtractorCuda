#include <iostream>
#include <vector>
#include <cmath>
#include "GrayPair.h"
#include "GLCM.h"
#include "WindowFeatureComputer.h"

int main() {
    std::cout << "Hello, World!" << std::endl;

    // Mockup Matrix
    int testData[4][4] = {{0,0,1,1},{1,0,1,1},{0,2,2,2},{2,2,3,3}};
    int rows = 4;
    int columns = 4;
    int maxGrayLevel = 4;
    int windowDimension = 4;

    vector<int> inputPixels(pow(windowDimension,2));
    cout << "Img = " << endl;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            cout << testData[i][j] << " " ;
            inputPixels[i * (windowDimension) + j] = testData[i][j];
        }
        cout << endl;
    }

    cout << endl << "Linearized Input matrix:" << endl;

    typedef vector<int>::const_iterator VI;
    for(VI element=inputPixels.begin(); element != inputPixels.end(); element++)
    {
        cout << *element << " " ;
    }
    cout << endl ;

    int distance = 1;
    // Start Creating the first GLCM
    // 4x4 0° 1 pixel distanza
    int shiftRows = 0;
    int shiftColumns = 1;

    GLCM glcm0(distance, shiftRows, shiftColumns, windowDimension, maxGrayLevel);
    glcm0.printGLCMData();
    glcm0.initializeElements(inputPixels);
    glcm0.printGLCMElements();
    glcm0.printAggregated();

    WindowFeatureComputer wfc;
    map<string, double> features = wfc.computeFeatures(glcm0);
    wfc.printFeatures(features);
    return 0;
}