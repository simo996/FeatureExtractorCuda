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
        WindowFeatureComputer(vector<int>& inputPixels);
        vector<map<string, double>> computeFeatures(); // 1 of each of the 4 dimensions
        void printFeaturesList(vector<map<string, double>>& featureList);
    private:
        int distance;
        int windowDimension;
};


#endif //FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H
