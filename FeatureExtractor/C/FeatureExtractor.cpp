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

struct GLCMData
{
	int distance;
	// Values necessary to identify neighbor pixel
	int shiftX;
	int shiftY;
	// Sub Borders in the windows according to direction
	int borderX;
	int borderY;
	int numberOfPairs;
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

bool testAddElements(int * metaGLCM, int metaGlcmLength, const int numberOfPairs, const int grayLevel)
{
	int sample[4]={48,144,122, 121};
	int * tempAdd = (int *) malloc(sizeof(int)*(4+metaGlcmLength));
	int temp = addElements(metaGLCM, sample, tempAdd,  metaGlcmLength, 4,numberOfPairs, grayLevel);
	// PER QUALCHE STRANO MOTIVO LA SECONDA COMPRESS NON RITORNA
	printArray(tempAdd, temp);
	//printMetaGlcm(tempAdd, temp, numberOfPairs, grayLevel);
	return true;
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

	window.rows = 4;
	window.columns = 4;

	imageMatrix = Mat(4,4,CV_32S,&testData);

	// Linearized matrix of pixels
	int * inputPixels = (int *) malloc(sizeof(int) * window.rows * window.columns);

	// Test To see if correctly loaded in MAT
	cout << "Img = " << endl;
	for (int i = 0; i < imgData.rows; i++)
	{
		for (int j = 0; j < imgData.columns; j++)
		{
			cout << imageMatrix.at<int>(i,j) <<" " ;
			inputPixels[i * (window.columns) + j] = imageMatrix.at<int>(i,j);
		}
		cout << endl;
	}
	// Start Creating the first GLCM
	// 4x4 0° 1 pixel distanza
	int distance = 1;
	int shiftY = 0;
	int shiftX = 1;

	GLCM glcm0x0;
	initializeMetaGLCM(&glcm0x0, distance, shiftX, shiftY, window.rows, window.columns);
	printGLCMData(glcm0x0);
	initializeMetaGLCMElements(&glcm0x0, inputPixels, imgData.grayLevel);

	// See the output
	cout << "Codified metaGlcm";
	printMetaGlcm(glcm0x0, imgData.grayLevel);

	double features[16];
    computeFeatures(features,glcm0x0, imgData.grayLevel);
	printFeatures(features);

	return 0;
}


