//
// Created by simo on 05/08/18.
//

#ifndef FEATUREEXTRACTOR_IMAGEREADER_H
#define FEATUREEXTRACTOR_IMAGEREADER_H

#include <iostream>
#include "Image.h"
#include <opencv/cv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

class ImageLoader {
public:
    static Mat readMriImage(string fileName);
    static Image readImage(const string fileName);
    static void showImage(Mat& img, string windowName);
    static void showImageStretched(Mat& img, string windowName);
    static void printMatImageData(Mat& img);
};


#endif //FEATUREEXTRACTOR_IMAGEREADER_H
