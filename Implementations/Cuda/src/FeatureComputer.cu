
#include "FeatureComputer.h"

__device__ FeatureComputer::FeatureComputer(const unsigned int * pixels, const ImageData& img,
        const int shiftRows, const int shiftColumns,
        const Window& wd, WorkArea& wa)
                                 : pixels(pixels), image(img),
                                   windowData(wd), workArea(wa) {
    // Each direction has 2 shift used for addressing each pixel
    windowData.setDirectionShifts(shiftRows, shiftColumns);
    
    /* Deduct what window this thread is computing for saving the results
     * in the right memory location */
    computeOutputWindowFeaturesIndex();
    int featuresCount = Features::getSupportedFeaturesCount();
    int actualWindowOffset = outputWindowOffset * featuresCount;  // consider space for each feature
    double * rightLocation = workArea.output + actualWindowOffset; // where results will be saved
    featureOutput = rightLocation;
    // Compute features
    computeDirectionalFeatures();
}

/* This method produces a value is the number of the window in the total
 * window set of the image*/
__device__ void FeatureComputer::computeOutputWindowFeaturesIndex(){
    // If bordered, the original image is at the center
    int rowOffset = windowData.imageRowsOffset - image.getBorderSize();
    int colOffset = windowData.imageColumnsOffset - image.getBorderSize();
    outputWindowOffset = (rowOffset * (image.getColumns() - 2 * image.getBorderSize()))
            + colOffset;
}

/* Computes all the features supported.
 * The results will be saved in the array of the work area given to this thread
 */
__device__ void FeatureComputer::computeDirectionalFeatures() {
    // Generate the 5 needed array of representations
    GLCM glcm(pixels, image, windowData, workArea);
    //glcm.printGLCM(); // Print data and elements for debugging

    // Features computable from glcm Elements
    extractAutonomousFeatures(glcm, featureOutput);

    // Feature computable from aggregated glcm pairs
    extractSumAggregatedFeatures(glcm, featureOutput);
    extractDiffAggregatedFeatures(glcm, featureOutput);

    // Imoc
    extractMarginalFeatures(glcm, featureOutput);
}


// ASM
__device__ inline double computeAsmStep(const double actualPairProbability){
    return pow((actualPairProbability),2);
}

// AUTOCORRELATION
__device__ inline double computeAutocorrelationStep(const uint i, const uint j, const double actualPairProbability){
    return (i * j * actualPairProbability);
}

// ENTROPY
__device__ inline double computeEntropyStep(const double actualPairProbability){
    return (actualPairProbability * log(actualPairProbability));
}

// HOMOGENEITY
__device__ inline double computeHomogeneityStep(const uint i, const uint j, const double actualPairProbability){
    int diff = i - j; // avoids casting value errors of uint(negative number)
    diff = diff < 0 ? -diff : diff; // absolute value
    return (actualPairProbability / (1 + diff));
}

// CONTRAST
__device__ inline double computeContrastStep(const uint i, const uint j, const double actualPairProbability){
    int diff = i - j; // avoids casting value errors of uint(negative number)
    diff = diff < 0 ? -diff : diff; // absolute value
    return (actualPairProbability * (pow(diff, 2)));
}

// DISSIMILARITY
__device__ inline double computeDissimilarityStep(const uint i, const uint j, const double pairProbability){
	int diff = i - j; // avoids casting value errors of uint(negative number)
    diff = diff < 0 ? -diff : diff; // absolute value
    return (pairProbability * diff);
}

// IDM
__device__ inline double computeInverceDifferenceMomentStep(const uint i, const uint j,
    const double pairProbability, const uint maxGrayLevel) {
	double diff = i - j; // avoids casting value errors of uint(negative number)
    return (pairProbability / (1 + fabs(diff) / maxGrayLevel));
}

/* FEATURES WITH MEANS */
// CORRELATION
__device__ inline double computeCorrelationStep(const uint i, const uint j,
    const double pairProbability, const double muX, const double muY,
    const double sigmaX, const double sigmaY){
    // beware ! unsigned int - double
    return (((i - muX) * (j - muY) * pairProbability ) / (sigmaX * sigmaY));
}

// CLUSTER PROMINENCE
__device__ inline double computeClusterProminenceStep(const uint i, const uint j,
    const double pairProbability, const double muX, const double muY){
    return (pow((i + j - muX - muY), 4) * pairProbability);
}

// CLUSTER SHADE
__device__ inline double computeClusterShadeStep(const uint i, const uint j,
    const double pairProbability, const double muX, const double muY){
    return (pow((i + j - muX - muY), 3) * pairProbability);
}

// SUM OF SQUARES
__device__ inline double computeSumOfSquaresStep(const uint i,
                                      const double pairProbability, const double mean){
    return (pow((i - mean), 2) * pairProbability);
}

// SUM Aggregated features
// SUM AVERAGE
__device__ inline double computeSumAverageStep(const double aggregatedGrayLevel, const double pairProbability){
    return (aggregatedGrayLevel * pairProbability);
}

// SUM ENTROPY
__device__ inline double computeSumEntropyStep(const double pairProbability){
    return (log(pairProbability) * pairProbability);
}

// SUM VARIANCE
__device__ inline double computeSumVarianceStep(const uint aggregatedGrayLevel,
    const double pairProbability, const double sumEntropy){
    // beware ! unsigned int - double
    return (pow((aggregatedGrayLevel - sumEntropy),2) * pairProbability);
}

// DIFF Aggregated features
// DIFF ENTROPY
__device__ inline double computeDiffEntropyStep(const double pairProbability){
    return (log(pairProbability) * pairProbability);
}

// DIFF
__device__ inline double computeDiffVarianceStep(const uint aggregatedGrayLevel, const double pairProbability){
    return ((aggregatedGrayLevel * aggregatedGrayLevel) * pairProbability);
}

// Marginal Features
__device__ inline double computeHxStep(const double grayLevelProbability){
    return (grayLevelProbability * log(grayLevelProbability));
}

__device__ inline double computeHyStep(const double grayLevelProbability){
    return (grayLevelProbability * log(grayLevelProbability));
}



/*
    This method will compute all the features computable from glcm gray level pairs
*/
__device__ void FeatureComputer::extractAutonomousFeatures(const GLCM& glcm, double* features){
    // Intermediate values
    double mean = 0;
    double muX = 0;
    double muY = 0;
    double sigmaX = 0;
    double sigmaY = 0;

    // Actual features
    double angularSecondMoment = 0;
    double autoCorrelation = 0;
    double entropy = 0;
    double maxprob = 0;
    double homogeneity = 0;
    double contrast = 0;
    double dissimilarity = 0;
    double idm = 0;

    // First batch of computable features
    int length = glcm.effectiveNumberOfGrayPairs;
    for (int k = 0; k < length; ++k) {
        GrayPair actualPair = glcm.grayPairs[k];

        grayLevelType i = actualPair.getGrayLevelI();
        grayLevelType j = actualPair.getGrayLevelJ();
        double actualPairProbability = ((double) actualPair.getFrequency())/glcm.getNumberOfPairs();

        angularSecondMoment += computeAsmStep(actualPairProbability);
        autoCorrelation += computeAutocorrelationStep(i, j, actualPairProbability);
        entropy += computeEntropyStep(actualPairProbability);
        if(maxprob < actualPairProbability)
            maxprob = actualPairProbability;
        homogeneity += computeHomogeneityStep(i, j, actualPairProbability);
        contrast += computeContrastStep(i, j, actualPairProbability);
        dissimilarity += computeDissimilarityStep(i, j, actualPairProbability);
        idm += computeInverceDifferenceMomentStep(i, j, actualPairProbability, glcm.getMaxGrayLevel());

        // intemediate values
        mean += (i * j * actualPairProbability);
        muX += (i * actualPairProbability);
        muY += (j * actualPairProbability);
    }

    features[ASM] = angularSecondMoment;
    features[AUTOCORRELATION] = autoCorrelation;
    features[ENTROPY] = (-1 * entropy);
    features[MAXPROB] = maxprob;
    features[HOMOGENEITY] = homogeneity;
    features[CONTRAST] = contrast;
    features[DISSIMILARITY] = dissimilarity;
    features[IDM] = idm;

    // Second batch of computable features
    double clusterProm = 0;
    double clusterShade = 0;
    double sumOfSquares = 0;

    for (int k = 0; k < length; ++k)
    {
        GrayPair actualPair = glcm.grayPairs[k];
        grayLevelType i = actualPair.getGrayLevelI();
        grayLevelType j = actualPair.getGrayLevelJ();
        double actualPairProbability = ((double) actualPair.getFrequency())/glcm.getNumberOfPairs();

        clusterProm += computeClusterProminenceStep(i, j, actualPairProbability, muX, muY);
        clusterShade += computeClusterShadeStep(i, j, actualPairProbability, muX, muY);
        sumOfSquares += computeSumOfSquaresStep(i, actualPairProbability, mean);
        sigmaX += pow((i - muX), 2) * actualPairProbability;
        sigmaY += pow((j - muY), 2) * actualPairProbability;
    }

    sigmaX = sqrt(sigmaX);
    sigmaY = sqrt(sigmaY);

    features[CLUSTERPROMINENCE] = clusterProm;
    features[CLUSTERSHADE] = clusterShade;
    features[SUMOFSQUARES] = sumOfSquares;

    // Only feature that needs the third scan of the glcm
    double correlation = 0;

    for (int k = 0; k < length; ++k)
    {
        GrayPair actualPair = glcm.grayPairs[k];
        grayLevelType i = actualPair.getGrayLevelI();
        grayLevelType j = actualPair.getGrayLevelJ();
        double actualPairProbability = ((double) actualPair.getFrequency())/glcm.getNumberOfPairs();

        correlation += computeCorrelationStep(i, j, actualPairProbability,
            muX, muY, sigmaX, sigmaY);
    }
    features[CORRELATION] = correlation;

}

/*
    This method will compute the 3 features obtained from the pairs <k, int freq>
    where k is the sum of the 2 gray leveles <i,j> in a pixel pair of the glcm
*/
__device__ void FeatureComputer::extractSumAggregatedFeatures(const GLCM& glcm, double* features) {
    int numberOfPairs = glcm.getNumberOfPairs();

    double sumavg = 0;
    double sumentropy = 0;
    double sumvariance = 0;

    // First batch of computable features
    int length = glcm.numberOfSummedPairs;
    for (int i = 0; i < length; ++i) {
        AggregatedGrayPair actualPair = glcm.summedPairs[i];
        grayLevelType k = actualPair.getAggregatedGrayLevel();
        double actualPairProbability = ((double) actualPair.getFrequency()) / numberOfPairs;

        sumavg += computeSumAverageStep(k, actualPairProbability);
        sumentropy += computeSumEntropyStep(actualPairProbability);
    }
    sumentropy *= -1;
    features[SUMAVERAGE] = sumavg;
    features[SUMENTROPY] = sumentropy;

    for (int i = 0; i < length; ++i) {
        AggregatedGrayPair actualPair = glcm.summedPairs[i];
        grayLevelType k = actualPair.getAggregatedGrayLevel();
        double actualPairProbability = ((double) actualPair.getFrequency()) / numberOfPairs;

        sumvariance += computeSumVarianceStep(k, actualPairProbability, sumentropy);
    }

    features[SUMVARIANCE] = sumvariance;
}

/*
    This method will compute the 2 features obtained from the pairs <k, int freq>
    where k is the absolute difference of the 2 gray leveles in a pixel pair
    <i,j> of the glcm
*/
__device__ void FeatureComputer::extractDiffAggregatedFeatures(const GLCM& glcm, double* features) {
    int numberOfPairs= glcm.getNumberOfPairs();

    double diffentropy = 0;
    double diffvariance = 0;

    int length = glcm.numberOfSubtractedPairs;
    for (int i = 0; i < length; ++i) {
        AggregatedGrayPair actualPair = glcm.subtractedPairs[i];
        grayLevelType k = actualPair.getAggregatedGrayLevel();
        double actualPairProbability = ((double) actualPair.getFrequency()) / numberOfPairs;

        diffentropy += computeDiffEntropyStep(actualPairProbability);
        diffvariance += computeDiffVarianceStep(k, actualPairProbability);
    }
    diffentropy *= -1;
    features[DIFFENTROPY] = diffentropy;
    features[DIFFVARIANCE] = diffvariance;

}

/*
    This method will compute the only feature computable from the "marginal
    representation" of the pairs <(X, ?), int frequency> and the pairs
    <(?, X), int frequency> of reference/neighbor pixel
*/
__device__ void FeatureComputer::extractMarginalFeatures(const GLCM& glcm, double* features){
    int numberOfPairs = glcm.getNumberOfPairs();
    double hx = 0;

    // Compute first intermediate value
    int xLength = glcm.numberOfxMarginalPairs;
    for (int k = 0; k < xLength; ++k) {
        double probability = ((double) (glcm.xMarginalPairs[k].getFrequency())/numberOfPairs);

        hx += computeHxStep(probability);
    }
    hx *= -1;

    // Compute second intermediate value
    double hy = 0;
    int yLength = glcm.numberOfyMarginalPairs;
    for (int k = 0; k < yLength; ++k) {
        double probability = ((double) (glcm.yMarginalPairs[k].getFrequency())/numberOfPairs);

        hy += computeHyStep(probability);
    }
    hy *= -1;

    // Extract third intermediate value
    double hxy = features[ENTROPY];

    // Compute last intermediate value
    double hxy1 = 0;

    int length = glcm.effectiveNumberOfGrayPairs;
    for (int l = 0; l < length; ++l) {
        GrayPair actualPair = glcm.grayPairs[l];
        double actualPairProbability = ((double) glcm.grayPairs[l].getFrequency()) / numberOfPairs;

        AggregatedGrayPair i (actualPair.getGrayLevelI(), 0); // 0 frequency is placeholder
        int xposition = 0;
        // it will be found, no need to check boundaries
        while((!glcm.xMarginalPairs[xposition].compareTo(i)) && (xposition < glcm.numberOfxMarginalPairs))
            xposition++;
        double xMarginalProbability = (double) glcm.xMarginalPairs[xposition].getFrequency() / numberOfPairs;

        AggregatedGrayPair j (actualPair.getGrayLevelJ(), 0); // 0 frequency is placeholder
        int yposition = 0;
        // it will be found, no need to check boundaries
        while((!glcm.yMarginalPairs[yposition].compareTo(j)) && (yposition < glcm.numberOfyMarginalPairs))
            yposition++;
        double yMarginalProbability = (double) glcm.yMarginalPairs[yposition].getFrequency() / numberOfPairs;

        hxy1 += actualPairProbability * log(xMarginalProbability * yMarginalProbability);
    }
    hxy1 *= -1;
    features[IMOC] = (hxy - hxy1)/(max(hx, hy));

}
