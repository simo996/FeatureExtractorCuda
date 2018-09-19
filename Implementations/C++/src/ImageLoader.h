#ifndef FEATUREEXTRACTOR_IMAGEREADER_H
#define FEATUREEXTRACTOR_IMAGEREADER_H

#include <iostream>
#include "ImageData.h"
#include <opencv/cv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

/** This class uses OpenCv to read, transform and save images allowing the tool
 * to work with every image, in color channels or grayscale, format supported
 * by openCv but without being highly coupled to it
*/
class ImageLoader {
public:
    /**
     * Method that external components will invoke to get an Image instance
     * @param fileName: the path/name of the image to read
     * @param borderType: type of the border to apply to the image read
     * @param borderSize: border to apply to each side of the image read
     * @param quantitize: reduction of grayLevels to apply to the image read
     * @param quantizationMax: maximum gray level when quantitization is
     * applied to reduce the graylevels in [0,quantizationMax]
     * @return
     */
    static Image readImage(string fileName, short int borderType, int borderSize, bool quantitize, int quantizationMax);
    /**
     * Method used when generating feature images with the features values computed
     * @param rows
     * @param cols
     * @param input: list of all the features values used as intensity in the
     * output image
     * @return image obtained from features values provided
     */
    static Mat createDoubleMat(int rows, int cols, const vector<double>& input);
    /**
     * Save the feature image on disk
     * @param image to save
     * @param fileName path where to save the image
     * @param stretch linear stretch applied to enhance quality with very
     * dark/bright images
     */
    static void saveImage(const Mat &image, const string &fileName,
                          bool stretch = true);
    // DEBUG method
    static void showImagePaused(const Mat& img, const string& windowName);
private:
    /**
     * Invocation of Opencv standard reading method from file system
     * @param fileName
     * @return
     */
    static Mat readImage(string fileName);
    /**
     * Converting images with colors to grayScale
     * @param inputImage
     * @return
     */
    static Mat convertToGrayScale(const Mat& inputImage);
    /**
     * Quantitze gray levels in set [0, Max]
     * @param inputImage
     * @param maxLevel
     * @return
     */
    static Mat quantitizeImage(Mat& inputImage, int maxLevel);
    /**
     * Returnes a stretched image to enhance details in over/under exposed
     * input
     * @param inputImage
     * @return
     */
    static Mat stretchImage(const Mat& inputImage);
    /**
     * Save the image on the file system
     * @param img
     * @param fileName
     */
    static void saveImageToFileSystem(const Mat& img, const string& fileName);
    /**
     * Add borders to the image read
     * @param img
     * @param borderType
     * @param borderSize
     */
    static void addBorderToImage(Mat &img, short int borderType, int borderSize);

};


#endif //FEATUREEXTRACTOR_IMAGEREADER_H