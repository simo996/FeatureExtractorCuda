//
// Created by simo on 16/07/18.
//

#ifndef FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H
#define FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H

#include "FeatureComputer.h"
#include "Direction.h"

typedef vector<vector<double>> WindowFeatures;
typedef vector<double> FeatureValues;

using namespace std;

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
private:
        // Initialization data to pass to each FeatureComputer
        WorkArea& workArea;
        unsigned int * pixels;
        ImageData image;
        Window windowData;
};


#endif //FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H
