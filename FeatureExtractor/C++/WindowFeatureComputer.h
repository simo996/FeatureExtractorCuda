//
// Created by simo on 16/07/18.
//

#ifndef FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H
#define FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H

#include <vector>
#include <map>
#include "FeatureComputer.h"

using namespace std;

struct FeatureBundle{
  string directionLabel;
  map<FeatureNames, double> features;
};

typedef vector<FeatureBundle> WindowFeatures;

class WindowFeatureComputer {
  /*
     * RESPONSABILITA CLASSE: Computare le feature per la finestra nelle 4 direzioni
     */
public:
        WindowFeatureComputer(const vector<int>& inputPixels, int maxGrayLevel, Window wd);
        WindowFeatures computeBundledFeatures(); // 1 of each of the 4 dimensions
        void printBundledFeatures(WindowFeatures featureList) const;

        // TODO think about wich of the alternative suits better the ImageFeatureComputer
        void printSeparatedFeatures(vector<map<FeatureNames, double>> featureList) const;
        vector<map<FeatureNames, double>> computeFeatures(); // 1 of each of the 4 dimensions

private:
        // Initialization data to pass to each FeatureComputer
        vector<int> inputPixels;
        int maxGrayLevel;
        Window windowData;
};


#endif //FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H
