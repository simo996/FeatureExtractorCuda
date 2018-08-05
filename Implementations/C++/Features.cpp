//
// Created by simo on 01/08/18.
//

#include "Features.h"

// Support method

vector<FeatureNames> Features::getAllSupportedFeatures() {
    vector<FeatureNames> output(18);
    for (int i = 0; i <= IMOC; ++i) {
        output[i] = static_cast<FeatureNames>(i);
    }
    return output;

}

vector<string> Features::getAllFeaturesFileNames() {
    // The right method will append the right extension to the filename
    vector<string> fileNames = {
            "ASM",
            "AUTOCORRELATION",
            "ENTROPY",
            "MAXPROB",
            "HOMOGENEITY",
            "CONTRAST",
            "CORRELATION",
            "CLUSTERPROMINENCE",
            "CLUSTERSHADE",
            "SUMOFSQUARES",
            "DISSIMILARITY",
            "IDM",
            "SUMAVERAGE",
            "SUMENTROPY",
            "SUMVARIANCE",
            "DIFFENTROPY",
            "DIFFVARIANCE",
            "IMOC"
    };
    return fileNames;
}

/*
    This method will print to screen just the entire list of features provided
*/
void Features::printAllFeatures(const map<FeatureNames, double>& features){
    cout << endl;
    // Autonomous
    cout << "ASM: \t" << features.at(ASM) << endl;
    cout << "AUTOCORRELATION: \t" << features.at(AUTOCORRELATION) << endl;
    cout << "ENTROPY: \t" << features.at(ENTROPY) << endl;
    cout << "MAXIMUM PROBABILITY: \t" << features.at(MAXPROB) << endl;
    cout << "HOMOGENEITY: \t" << features.at(HOMOGENEITY) << endl;
    cout << "CONTRAST: \t" << features.at(CONTRAST) << endl;
    cout << "DISSIMILARITY: \t" << features.at(DISSIMILARITY) << endl;
    cout << "CORRELATION: \t" << features.at(CORRELATION) << endl;
    cout << "CLUSTER Prominence: \t" << features.at(CLUSTERPROMINENCE) << endl;
    cout << "CLUSTER SHADE: \t" << features.at(CLUSTERSHADE) << endl;
    cout << "SUM OF SQUARES: \t" << features.at(SUMOFSQUARES) << endl;
    cout << "IDM normalized: \t" << features.at(IDM) << endl;
    // Sum aggregated
    cout << "SUM AVERAGE: \t" << features.at(SUMAVERAGE) << endl;
    cout << "SUM ENTROPY: \t" << features.at(SUMENTROPY) << endl;
    cout << "SUM VARIANCE: \t" << features.at(SUMVARIANCE) << endl;
    // Diff Aggregated
    cout << "DIFF ENTROPY: \t" << features.at(DIFFENTROPY) << endl;
    cout << "DIFF VARIANCE: \t" << features.at(DIFFVARIANCE) << endl;
    // Marginal
    cout << "INFORMATION MEASURE OF CORRELATION: \t" << features.at(IMOC) << endl;

    cout << endl;
}


/*
    This method will perform "a conversion ENUM->String"
*/
string Features::getFeatureName(FeatureNames featureName){
    switch(featureName){
        case (ASM):
            return "ASM: ";
        case (AUTOCORRELATION):
            return "AUTOCORRELATION: " ;
        case (ENTROPY):
            return "ENTROPY: ";
        case (MAXPROB):
            return "MAXIMUM PROBABILITY: ";
        case (HOMOGENEITY):
            return "HOMOGENEITY: ";
        case (CONTRAST):
            return "CONTRAST: ";
        case (DISSIMILARITY):
            return "DISSIMILARITY: ";
        case (CORRELATION):
            return "CORRELATION: ";
        case (CLUSTERPROMINENCE):
            return "CLUSTER Prominence: ";
        case (CLUSTERSHADE):
            return "CLUSTER SHADE: ";
        case (SUMOFSQUARES):
            return "SUM OF SQUARES: " ;
        case (SUMAVERAGE):
            return "SUM AVERAGE: ";
        case (IDM):
            return "IDM normalized: ";
        case (SUMENTROPY):
            return "SUM ENTROPY: ";
        case (SUMVARIANCE):
            return "SUM VARIANCE: ";
        case (DIFFENTROPY):
            return "DIFF ENTROPY: ";
        case (DIFFVARIANCE):
            return "DIFF VARIANCE: ";
        case (IMOC):
            return "INFORMATION MEASURE OF CORRELATION: ";
        default:
            fputs("Fatal Error! Unrecognized feature", stderr);
            exit(-1);
    }
}


/*
    This method will print to screen the ENUM label
*/
void Features::printFeatureName(FeatureNames featureName){
    cout << getFeatureName(featureName);
}

void Features::printSingleFeature(const map<FeatureNames, double>& features,
                                         FeatureNames featureName){

    typedef map<FeatureNames, double>::const_iterator MI;
    for (MI element = features.end(); element != features.end(); ++element)
    {
        // Print the label with the apposite method
        printFeatureName(element->first);
        // Print the value
        cout << element->second << endl;
    }
}