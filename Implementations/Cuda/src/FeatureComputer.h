//
// Created by simo on 11/07/18.
//

#ifndef FEATUREEXTRACTOR_FEATURECOMPUTER_H
#define FEATUREEXTRACTOR_FEATURECOMPUTER_H

#include <vector>
#include "GLCM.h"
#include "Features.h"

class FeatureComputer {
    /*
     * RESPONSABILITA CLASSE: Computare le 18 features per la singola direzione della finestra
     * Espone metodi per stampare i risultati
     */

public:
    FeatureComputer(const unsigned int * pixels, const ImageData& img,
            int shiftRows, int shiftColumns, const Window& windowData,
            WorkArea& wa, short int directionNumber);
    void computeDirectionalFeatures();
private:
    // given data to initialize related GLCM
    const unsigned int * pixels;
    ImageData image;
    Window windowData;
    WorkArea& workArea;
    // offset to indentify where to put results
    short int directionOffset;
    int outputWindowOffset;

    // Actual computation of all 18 features
    void computeBatchFeatures(const GLCM& metaGLCM);
    void extractAutonomousFeatures(const GLCM& metaGLCM, vector<double>& features);
    void extractSumAggregatedFeatures(const GLCM& metaGLCM, vector<double>& features);
    void extractDiffAggregatedFeatures(const GLCM& metaGLCM, vector<double>& features);
    void extractMarginalFeatures(const GLCM& metaGLCM, vector<double>& features);

    void computeOutputWindowFeaturesIndex();
    // Support method useful for debugging this class
    static void printGLCM(const GLCM& glcm); // prints glcms various information
};


#endif //FEATUREEXTRACTOR_FEATURECOMPUTER_H
