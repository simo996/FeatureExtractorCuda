
#ifndef IMAGEFEATURECOMPUTER_H_
#define IMAGEFEATURECOMPUTER_H_

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include "ImageLoader.h"
#include "ProgramArguments.h"
#include "Utils.h"
#include "WindowFeatureComputer.h"

/**
 * Array of all the features that can be extracted simultaneously from a
 * window
 */
typedef vector<double> WindowFeatures;
/**
 * Array of all the features that can be extracted simultaneously from a
 * direction in a window
 */
typedef vector<double> FeatureValues;

using namespace cv;

/**
 * This class has 3 main tasks:
 * - Read and transform the image according to the options provided
 * - Compute all the features in all the windows that can be created in the
 * image according to the options provided
 * - Organizing and saving the results
 */
class ImageFeatureComputer {
public:
	/**
	 * Initialize the class
	 * @param progArg: parameters of the problem
	 */
	ImageFeatureComputer(const ProgramArguments& progArg);
	/**
	 * This method will read the image, compute the features, re-arrange the
	 * results and save them as need on the file system
	 */
	void compute();
	/**
     * This method will compute all the features for every window for the
     * number of directions provided
     * @param pixels: pixels intensities of the image provided
     * @param img: image metadata
     * @return array (1 for each window) of array (1 for each computed direction)
     * of array of doubles (1 for each feature)
     */
    vector<vector<WindowFeatures>> computeAllFeatures(unsigned int * pixels, const ImageData& img);

    // EXTRAPOLATING RESULTS
	/**
     * This method will extract the results from each window
     * @param imageFeatures: array (1 for each window) of array (1 for each
     * computed direction) of array of doubles (1 for each feature)
     * @return array (1 for each direction) of array (1 for each feature) of all
     * the values computed of that feature
     * Es. <Entropy , (0.1, 0.2, 3, 4 , ...)>
     */
    vector<vector<vector<double>>> getAllDirectionsAllFeatureValues(const vector<vector<WindowFeatures>>& imageFeatures);

	// SAVING RESULTS ON FILES
	/**
	 * This method will save on different folders, all the features values
	 * computed for each directions of the image
	 * @param imageFeatures
	 */
    void saveFeaturesToFiles(const vector<vector<vector<double>>>& imageFeatures);

    // IMAGING
    /**
     * This method will produce and save all the images associated with each feature
     * for each direction
     * @param rowNumber: how many rows each image will have
     * @param colNumber: how many columns each image will have
     * @param imageFeatures
     */
    void saveAllFeatureImages(int rowNumber,  int colNumber, const vector<vector<FeatureValues>>& imageFeatures);


private:
	ProgramArguments progArg;

	// SUPPORT FILESAVE methods
	/**
	 * This method will save into the given folder, alle the values of all
	 * the features computed for 1  directions
	 * @param imageDirectedFeatures: all the values computed for each feature
	 * in 1 direction of the image
	 * @param outputFolderPath
	 */
	void saveDirectedFeaturesToFiles(const vector<vector<double>>& imageDirectedFeatures,
			const string& outputFolderPath);
	/**
	 * This method will save into the given folder, all the values for 1 feature
     * computed for 1 directions
	 * @param imageFeatures all the feature values of 1 feature
	 * @param path
	 */
	void saveFeatureToFile(const pair<FeatureNames, vector<double>>& imageFeatures, const string path);

	// SUPPORT IMAGING methods
	/**
	 * This method will produce and save all the images associated with
	 * each feature in 1 direction
	 * @param rowNumber: how many rows each image will have
	 * @param colNumber: how many columns each image will have
	 * @param imageFeatures: all the values computed for each feature of the image
	 * @param outputFolderPath: where to save the image
	 */
	void saveAllFeatureDirectedImages(int rowNumber,  int colNumber,
			const vector<vector<double>> &imageFeatures, const string& outputFolderPath);
	/**
	 * This method will produce and save on the filesystem the image associated with
	 * a feature in 1 direction
	 * @param rowNumber: how many rows the image will have
	 * @param colNumber: how many columns the image will have
	 * @param featureValues: values that will be the intensities values of the
	 * image
	 * @param outputFilePath: where to save the image
	 */
	void saveFeatureImage(int rowNumber,  int colNumber,
			const vector<double>& featureValues, const string& outputFilePath);
	
	/**
	 * Utility method
	 * @return applied border to the original image read
	 */
	int getAppliedBorders();
	/**
	 * Display a set of information about the computation of the provided image
	 * @param imgData
	 * @param padding
	 */
    void printInfo(ImageData imgData, int padding);
	/**
	 * Display the memory space used while computing the problem
	 * @param imgData
	 * @param padding
	 */
	void printExtimatedSizes(const ImageData& img);
};



#endif /* IMAGEFEATURECOMPUTER_H_ */
