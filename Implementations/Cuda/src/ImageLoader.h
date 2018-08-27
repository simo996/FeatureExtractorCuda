/*
 * ImageLoader.h
 *
 *  Created on: 26/ago/2018
 *      Author: simone
 */

#ifndef IMAGELOADER_H_
#define IMAGELOADER_H_


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
    static Image readImage(string fileName, bool cropResolution);
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


#endif /* IMAGELOADER_H_ */
