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
    This method will print to screen just the given feature
    performing "a conversion ENUM->String"
*/
// TODO think about moving closer to enum definition
void Features::printFeatureName(FeatureNames featureName){
    switch(featureName){
        case (ASM):
            cout << "ASM: \t";
            break;
        case (AUTOCORRELATION):
            cout << "AUTOCORRELATION: \t" ;
            break;
        case (ENTROPY):
            cout << "ENTROPY: \t";
            break;
        case (MAXPROB):
            cout << "MAXIMUM PROBABILITY: \t";
            break;
        case (HOMOGENEITY):
            cout << "HOMOGENEITY: \t";
            break;
        case (CONTRAST):
            cout << "CONTRAST: \t";
            break;
        case (DISSIMILARITY):
            cout << "DISSIMILARITY: \t";
            break;
        case (CORRELATION):
            cout << "CORRELATION: \t";
            break;
        case (CLUSTERPROMINENCE):
            cout << "CLUSTER Prominence: \t";
            break;
        case (CLUSTERSHADE):
            cout << "CLUSTER SHADE: \t";
            break;
        case (SUMOFSQUARES):
            cout << "SUM OF SQUARES: \t" ;
            break;
        case (SUMAVERAGE):
            cout << "SUM AVERAGE: \t";
            break;
        case (IDM):
            cout << "IDM normalized: \t";
            break;
        case (SUMENTROPY):
            cout << "SUM ENTROPY: \t";
            break;
        case (SUMVARIANCE):
            cout << "SUM VARIANCE: \t";
            break;
        case (DIFFENTROPY):
            cout << "DIFF ENTROPY: \t";
            break;
        case (DIFFVARIANCE):
            cout << "DIFF VARIANCE: \t";
            break;
        case (IMOC):
            cout << "INFORMATION MEASURE OF CORRELATION: \t";
            break;
        default:
            fputs("Fatal Error! Unrecognized direction", stderr);
            exit(-1);
    }
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