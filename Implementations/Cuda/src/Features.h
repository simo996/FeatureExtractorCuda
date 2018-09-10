#ifndef FEATURES_H_
#define FEATURES_H_

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__ 
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif

#include <vector>
#include <string>
#include <iostream>

using namespace std;

/*
 * List of all the feautures supported
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

/*
 * Helper class that lists all supported features and offers some
 * utilities methods about them
 */

class Features {
public:
	// used for allocating features array
	CUDA_HOSTDEV static int getSupportedFeaturesCount();
	CUDA_HOST static void printFeatureName(FeatureNames featureName);
	// return a list of the 18 features
	CUDA_HOST static vector<FeatureNames> getAllSupportedFeatures();
	// return a list of all the file names associated at features
	CUDA_HOST static vector<string> getAllFeaturesFileNames();
	// print features labels and their values
	CUDA_HOST static void printAllFeatures(const vector<double>& features);
	// print single feature label and its value
	CUDA_HOST static void printSingleFeature(const vector<double> &features,
	                                   FeatureNames featureName);
	CUDA_HOST static string printFeatureNameAndValue(double value, FeatureNames fname);

	// print the label associated with the enum
	CUDA_HOST static string getFeatureName(FeatureNames featureName);

};

#endif /* FEATURES_H_ */
