/*
 * Helper class that lists all supported features and offers some
 * utilities methods about them
 */
// Created by simo on 01/08/18.
//

#ifndef FEATUREEXTRACTOR_FEATURES_H
#define FEATUREEXTRACTOR_FEATURES_H

#include <vector>
#include <map>
#include <string>
#include <iostream>

using namespace std;

enum FeatureNames { ASM, AUTOCORRELATION, ENTROPY, MAXPROB, HOMOGENEITY, CONTRAST,
    CORRELATION, CLUSTERPROMINENCE, CLUSTERSHADE, SUMOFSQUARES, DISSIMILARITY, IDM,
    SUMAVERAGE, SUMENTROPY, SUMVARIANCE, DIFFENTROPY, DIFFVARIANCE, IMOC
};


class Features {
public:
    // return a list of the 18 features
    static vector<FeatureNames> getAllSupportedFeatures();
    // return a list of all the file names associated at features
    static vector<string> getAllFeaturesFileNames();
    // print features labels and their values
    static void printAllFeatures(const vector<double>& features);
    // print single feature label and its value
    static void printSingleFeature(const vector<double> &features,
                                   FeatureNames featureName);
    static string printFeatureNameAndValue(double value, FeatureNames fname);

        // print the label associated with the enum
    static void printFeatureName(FeatureNames featureName);
    static string getFeatureName(FeatureNames featureName);
};


#endif //FEATUREEXTRACTOR_FEATURES_H
