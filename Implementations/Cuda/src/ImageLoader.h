#ifndef IMAGELOADER_H_
#define IMAGELOADER_H_

#include <iostream>
#include "ImageData.h"
#include <opencv/cv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;


/* This class uses OpenCv to read, transform and save images allowing the tool
 * to work with every image, in color channels or grayscale, format supported
 * by openCv but without being highly coupled to it
*/

class ImageLoader {
public:
    // Method that external components will invoke to get an Image instance
    static Image readImage(string fileName, short int borderType, int borderSize, bool quantitize, int quantizationMax);
    // Method used when generating feature images with the features values computed
    static Mat createDoubleMat(int rows, int cols, const vector<double>& input);
    // Save the feature image on disk
    static void saveImage(const Mat &img, const string &fileName,
                          bool stretch = true);
    // DEBUG method
    static void showImagePaused(const Mat& img, const string& windowName);
private:
    // Opencv standard reading method from file system
    static Mat readImage(string fileName);
    // Converting images with colors to grayScale
    static Mat convertToGrayScale(const Mat& inputImage);
    // Quantitze gray levels in set [0, Max]
    static Mat quantitizeImage(Mat& inputImage, int maxLevel);
    // Images are stretched to enhance details in over/under exposed images
    static Mat stretchImage(const Mat& inputImage);
    // Save on the file system
    static void saveImageToFileSystem(const Mat& img, const string& fileName);
    // Custom padding to the image
    static void addBorderToImage(Mat &img, short int borderType, int borderSize);

};


#endif /* IMAGELOADER_H_ */
