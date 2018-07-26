#include <iostream>
#include <vector>
#include <cmath>
#include "WindowFeatureComputer.h"

int main() {
    std::cout << "Feature Extractor" << std::endl;

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
    // Start Creating the GLCMs
    Window wData(windowDimension, distance);
    WindowFeatureComputer fcw(inputPixels, maxGrayLevel, wData);
    WindowFeatures fs= fcw.computeBundledFeatures();
    fcw.printBundledFeatures(fs);

    return 0;
}