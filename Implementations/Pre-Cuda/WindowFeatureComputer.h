//
// Created by simo on 16/07/18.
//

#ifndef FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H
#define FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H

#include <vector>
#include <map>
#include "FeatureComputer.h"
#include "Direction.h"

using namespace std;

typedef vector<vector<double>> WindowFeatures; // will contain result for 4 directions
typedef vector<double> FeatureValues;

class WindowFeatureComputer {
    /*
   * RESPONSABILITA CLASSE: Computare le feature per la finestra nelle 4 direzioni
     * Fornire un stream di rappresentazione verso file
   */

public:
    WindowFeatureComputer(unsigned int * pixels, const ImageData& img, const Window& wd, WorkArea& wa);
    // Will be computed features in the directions specified
    // Default = 4 = all feautures ; oder 0->45->90->135°
    void computeWindowFeatures();
    /* Oss. No sense in computing a single feature, simply select the one
      needed from the complete list
     */
    // TODO return an ostream
    static void printAllDirectionsAllFeatures(const WindowFeatures &featureList); // 4 directions with dir label
    // Print the label of the direction and the features
    static void printSingleDirectionAllFeatures(const vector<double>& featureList); // with dir

private:
        // Initialization data to pass to each FeatureComputer
        WorkArea& workArea;
        unsigned int * pixels;
        ImageData image;
        Window windowData;
};


#endif //FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H
