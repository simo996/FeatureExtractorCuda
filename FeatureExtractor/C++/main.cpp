#include <iostream>
#include <vector>
#include <cmath>
#include "WindowFeatureComputer.h"

int main() {
    std::cout << "Feature Extractor" << std::endl;

    // Mockup Matrix
    int testData[] = {0,0,1,1,1,0,1,1,0,2,2,2,2,2,3,3};
    // Load Image object
    int *image = testData;
    int rows = 4;
    int columns = 4;
    int maxGrayLevel = 4;
    Image img(image, rows, columns, maxGrayLevel);
    img.printElements();
    // Load uniform window information (extracted from parameters
    int distance = 1;
    int windowDimension = 4;
    Window wData(windowDimension, distance);

    // DEBUG manually create the window
    wData.setSpacialOffsets(0,0);

    // Start Creating the GLCMs
    WindowFeatureComputer fcw(img, wData);
    WindowFeatures fs= fcw.computeBundledFeatures();
    fcw.printBundledFeatures(fs);

    return 0;
}