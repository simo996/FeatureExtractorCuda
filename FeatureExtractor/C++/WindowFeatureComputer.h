//
// Created by simo on 16/07/18.
//

#ifndef FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H
#define FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H

#include <vector>
#include <map>
#include "FeatureComputer.h"

using namespace std;

// Struct that represent all the supported directions
typedef struct Direction{
    string label;
    int shiftRows;
    int shiftColumns;
} Direction;

vector<Direction> getAllDirections();

// Commodity structs that encapsulate the feature list computed for 1 direction
struct FeatureBundle{
  string directionLabel;
  map<FeatureNames, double> features;
};
typedef vector<FeatureBundle> WindowFeatures; // will contain result for 4 directions


class WindowFeatureComputer {
    /*
   * RESPONSABILITA CLASSE: Computare le feature per la finestra nelle 4 direzioni
     * Fornire un stream di rappresentazione verso file
   */

public:
    WindowFeatureComputer(const Image& img, const Window& wd);
    // Will be computed features in the directions specified
    // Default = 4 = all feautures ; oder 0->45->90->135°
    WindowFeatures computeWindowFeatures(int numberOfDirections = 4);
    /* Oss. No sense in computing a single feature, simply select the one
      needed from the complete list
     */
    // TODO return an ostream
    static void printAllDirectionsFeatures(const WindowFeatures &featureList); // 4 directions with dir label
    static void printAllDirectionsSingleFeature(const WindowFeatures &featureList, FeatureNames featureName);
    static void printSingleDirectionAllFeatures(const FeatureBundle& featureList); // with dir
    static void printSingleDirectionSingleFeature(const FeatureBundle& featureList,  FeatureNames featureName);

private:
        // Initialization data to pass to each FeatureComputer
        Image image;
        Window windowData;
};


#endif //FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H
