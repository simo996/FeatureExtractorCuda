
#ifndef FEATUREEXTRACTOR_FEATURES_H
#define FEATUREEXTRACTOR_FEATURES_H

#include <vector>
#include <string>
#include <iostream>

using namespace std;


/**
 * List of all the features supported
 * The index in the enumeration is used for accessing the right cell
 * when saving results in a feature array
 */
enum FeatureNames {
    ASM,
    AUTOCORRELATION,
    ENTROPY,
    MAXPROB,
    HOMOGENEITY,
    CONTRAST,
    CORRELATION,
    CLUSTERPROMINENCE,
    CLUSTERSHADE,
    SUMOFSQUARES,
    DISSIMILARITY,
    IDM,
    // Sum Aggregated
    SUMAVERAGE,
    SUMENTROPY,
    SUMVARIANCE,
    // Diff Aggregated
    DIFFENTROPY,
    DIFFVARIANCE,
    // Marginal probability feature
    IMOC
};


/**
 * Helper class that lists all supported features and offers some
 * utilities methods about them
 */
class Features {
public:
    /**
     * Return a list of all the features supported
     * @return list of all the features supported
     */
    static vector<FeatureNames> getAllSupportedFeatures();
    /**
     * return a list of all the file names associated at features
     * @return list of all the file names associated at features
     */
    static vector<string> getAllFeaturesFileNames();
    /**
     * The quantity of features supported by this tool; used for allocating
     * arrays of features
     * @return quantity of features supported by this tool
     */
    static int getSupportedFeaturesCount();
    /**
     * DEBUG METHOD. This method will print features labels and their values
     * @param features
     */
    static void printAllFeatures(const vector<double>& features);
    /**
     * DEBUG METHOD. This method will print single feature label
     * @param features list of features computed
     * @param featureName index of the feature in the enumeration
     */
    static void printSingleFeature(const vector<double> &features,
                                   FeatureNames featureName);
    /**
     * DEBUG METHOD. This method will print single feature label and its value
     * @param value: value of the feature to print
     * @param fname: index of the feature in the enumeration
     * @return
     */
    static string printFeatureNameAndValue(double value, FeatureNames fname);
    /**
     * Print the label associated with the enum
     * @param featureName whose label will be printed
     */
    static void printFeatureName(FeatureNames featureName);
    /**
     * Returns as a string the label associated with the enum
     * @param featureName whose label will be returned
     */
    static string getFeatureName(FeatureNames featureName);
};


#endif //FEATUREEXTRACTOR_FEATURES_H
