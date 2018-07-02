#include "FeatureComputation.h"
#include "GrayPair.h"
#include "MetaGLCM.h"
#include "SupportCode.h"

// C libraries
#include <stdio.h>
#include <stdlib.h>
#include <limits.h> // see Max_int
#include <unistd.h> // Command Options
#include <math.h> 

// File c++ libraries
#include <fstream>
#include <iostream>
#include <iterator>
#include <algorithm> 
// OpenCv Libraries for loading MRImages
#include <opencv2/opencv.hpp>

using namespace cv; // Loading MRI images
using namespace std;

struct ImageData{
	int rows;
	int columns;
	int grayLevel; // 16_U, 16_S
};

struct WindowData{
	int rows;
	int columns;
};


void readMRImage(Mat image, struct ImageData imgData, const char * filename)
{
	image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE );
	if(!image.data){
		fprintf(stderr, "Error while opening the file\n");
		exit(-1);
	}
	else
	{	
		imgData.rows = image.rows;
		imgData.columns = image.cols;
		if(image.channels() == 1)
			imgData.grayLevel = image.depth();	
		else 
		{
			fprintf(stderr, "NOT a grayscale medical image\n");
			exit(-1);
		}
	}
}

void readFile(const char *filename, int *dataOut)
{
	ifstream inputFile;
	inputFile.open(filename, ios::in);
	if(!inputFile.good())
	{
		fprintf(stderr, "Error while opening the file\n");
		exit(-1);
	}
	else
	{
		// read data
		inputFile.close();
	}
}

void processMetaGLCM(struct GLCM * metaGLCM, const int * inputPixels)
{
	initializeMetaGLCMElements(metaGLCM, inputPixels);

	// See metadata
	cout << "\nMetadata on GLCM" ;
	printGLCMData(*metaGLCM);
	// See the output
	cout << "\nCodified metaGlcm";
	printMetaGlcm(*metaGLCM);

	double features[17];
	computeFeatures(features,*metaGLCM);
	printFeatures(features);
}

// TEST METHOD
void computeSomeGLCMs(int * inputPixels, int grayLevel)
{
	int windowDimension = 4;
	int distance = 1;
	
	// Start Creating the first GLCM
	// 4x4 0° 1 pixel distanza
	int shiftRows = 0;
	int shiftColumns = 1;

	GLCM glcm0;
	initializeMetaGLCM(&glcm0, distance, shiftRows, shiftColumns, windowDimension, grayLevel);
	processMetaGLCM(&glcm0, inputPixels);

	// Start Creating the second GLCM
	// 4x4 90° 1 pixel distanza
	shiftRows = -1;
	shiftColumns = 0;

	GLCM glcm90;
	initializeMetaGLCM(&glcm90, distance, shiftRows, shiftColumns, windowDimension, grayLevel);
	processMetaGLCM(&glcm90, inputPixels);

	// Start Creating the third GLCM
	// 4x4 45° 1 pixel distanza
	shiftRows = -1;
	shiftColumns = 1;

	GLCM glcm45;
	initializeMetaGLCM(&glcm45, distance, shiftRows, shiftColumns, windowDimension, grayLevel);
	processMetaGLCM(&glcm45, inputPixels);

	/* TODO FIX BORDELLO NELL'HEAP, CHE CAUSA LA MORTE DI QUESTO
	// Start Creating the third GLCM
	// 4x4 135° 1 pixel distanza
	shiftRows = -1;
	shiftColumns = -1;

	GLCM glcm135;
	initializeMetaGLCM(&glcm135, distance, shiftRows, shiftColumns, windowDimension, grayLevel);
	processMetaGLCM(&glcm135, inputPixels);
    */
}

int main(int argc, char const *argv[])
{
	Mat imageMatrix; // Matrix representation of the image
	ImageData imgData; // MetaData about the image
	WindowData window;
	//initialControls(argc,argv);  TODO Analyze given options

	/* read image and extract metadata
	readMRImage(imageMatrix,imgData, argv[1]);

	*/

	// Mockup Matrix
	int testData[4][4] = {{0,0,1,1},{1,0,1,1},{0,2,2,2},{2,2,3,3}};
	imgData.rows = 4;
	imgData.columns = 4;
	imgData.grayLevel = 4;

	int windowDimension = 4;

	imageMatrix = Mat(4,4,CV_32S,&testData);

	// Linearized matrix of pixels
	int * inputPixels = (int *) malloc(sizeof(int) * pow(windowDimension,2));

	// Test To see if correctly loaded in MAT
	cout << "Img = " << endl;
	for (int i = 0; i < imgData.rows; i++)
	{
		for (int j = 0; j < imgData.columns; j++)
		{
			cout << imageMatrix.at<int>(i,j) << " " ;
			inputPixels[i * (windowDimension) + j] = imageMatrix.at<int>(i,j);
		}
		cout << endl;
	}

	// 4x4 135° 1 pixel distanza
	int distance = 1;
	int shiftRows = -1;
	int shiftColumns = -1;

	/*
	GLCM glcm135;
	initializeMetaGLCM(&glcm135, distance, shiftRows, shiftColumns, windowDimension, imgData.grayLevel);
    initializeMetaGLCMElements(&glcm135, inputPixels);
    cout << "\nMetadata on GLCM" ;
    printGLCMData(&glcm135);
    // See the output
    cout << "\nCodified metaGlcm";
    printMetaGlcm(glcm135);

    double features[17];
    computeFeatures(features, glcm135);
    printFeatures(features);

    */
	// compute Other GLCMS
	computeSomeGLCMs(inputPixels, imgData.grayLevel);

	return 0;
}


