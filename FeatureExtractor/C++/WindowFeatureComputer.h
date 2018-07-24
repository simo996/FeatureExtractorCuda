//
// Created by simo on 16/07/18.
//

#ifndef FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H
#define FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H

#include <vector>
#include <map>

using namespace std;

class WindowFeatureComputer {
  /*
     * RESPONSABILITA CLASSE: Computare le feature per la finestra nelle 4 direzioni
     */
    public:
        WindowFeatureComputer(const vector<int>& inputPixels, int maxGrayLevel, int distance,
                              int windowDimension, bool symmetric = false);
        vector<map<string, double>> computeFeatures(); // 1 of each of the 4 dimensions
        void printSeparatedFeatures(vector<map<string, double>> featureList) const;
    private:
        vector<int> inputPixels;
        int maxGrayLevel;
        bool simmetric;
        int distance;
        int windowDimension;
};


#endif //FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H
