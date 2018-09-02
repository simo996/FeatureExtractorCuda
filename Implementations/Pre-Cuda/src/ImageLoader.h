//
// Created by simo on 05/08/18.
//

#ifndef FEATUREEXTRACTOR_IMAGEREADER_H
#define FEATUREEXTRACTOR_IMAGEREADER_H

#include <iostream>
#include "ImageData.h"
#include <opencv/cv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

class ImageLoader {
public:
    static Mat readMriImage(string fileName, bool cropResolution);
    static Image readImage(string fileName, bool cropResolution, int borderSize);
    static Mat createDoubleMat(int rows, int cols, const vector<double>& input);
    static Mat convertToGrayScale(const Mat& inputImage);
    static Mat concatenateStretchImage(const Mat& inputImage);
    static Mat stretchImage(const Mat& inputImage);
    static void stretchAndSave(const Mat &img, const string &fileName);
    static void saveImageToFile(const Mat& img, const string& fileName);
    static void showImage(const Mat& img, const string& windowName);
    static void showImagePaused(const Mat& img, const string& windowName);
    static void showImageStretched(const Mat& img, const string& windowName);
    static void printMatImageData(const Mat& img);
};


#endif //FEATUREEXTRACTOR_IMAGEREADER_H
