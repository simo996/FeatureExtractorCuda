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
    WindowFeatureComputer(const Image& img, const Window& wd);
    // Will be computed features in the directions specified
    // Default = 4 = all feautures ; oder 0->45->90->135°
    WindowFeatures computeWindowFeatures();
    /* Oss. No sense in computing a single feature, simply select the one
      needed from the complete list
     */
private:
        // Initialization data to pass to each FeatureComputer
        Image image;
        Window windowData;
};


#endif //FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H
