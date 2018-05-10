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
	printMetaGlcm(tempAdd, temp, numberOfPairs, grayLevel);
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
	imageMatrix = Mat(4,4,CV_32S,&testData);

	// Test To see if correctly loaded in MAT
	cout << "Img = " << endl;
	for (int i = 0; i < imgData.rows; i++)
	{
		for (int j = 0; j < imgData.columns; j++)
		{
			cout << imageMatrix.at<int>(i,j) <<" " ;
		}
		cout << endl;
	}
	window.rows = 4;
	window.columns = 4;
	// Start Creating the first GLCM
	// 4x4 0° 1 pixel distanza
	GLCMData glcm0;
	glcm0.distance = 1;
	glcm0.shiftY = 0;
	glcm0.shiftX = 1;
	// TODO change dimensions to reflect windows, not image
	glcm0.borderX = (window.columns - (glcm0.distance * glcm0.shiftX));
	glcm0.borderY = (window.rows - (glcm0.distance * glcm0.shiftY));
	int numberOfPairs = glcm0.borderX * glcm0.borderY;
	assert(numberOfPairs == 12);

	// Generation of the metaGLCM
	int * codifiedMatrix =(int *) malloc(sizeof(int)*numberOfPairs);
	int k = 0;
	int referenceGrayLevel;
	int neighborGrayLevel;

	// FIRST STEP: codify all pairs
	for (int i = 0; i < glcm0.borderY ; i++)
	{
		for (int j = 0; j < glcm0.borderX; j++)
		{
			referenceGrayLevel = imageMatrix.at<int>(i,j);
			neighborGrayLevel = imageMatrix.at<int>(i+glcm0.shiftY,j+glcm0.shiftX);

			codifiedMatrix[k] = (((referenceGrayLevel*imgData.grayLevel) +
			neighborGrayLevel) * (numberOfPairs)) ;
			k++;
		}
	}

	// See the output
	cout << "Codified metaGlcm";
	printArray(codifiedMatrix,numberOfPairs);

	// SECOND STEP: Order
	//int orderedCodifiedMatrix[numberOfPairs];
	sort(codifiedMatrix, numberOfPairs);
	cout << endl << "Ordered Codified metaGlcm";
	printArray(codifiedMatrix,numberOfPairs);

	// THIRD STEP: Compress
	int metaGlcmLength = localCompress(codifiedMatrix, numberOfPairs);

	int metaGLCM[metaGlcmLength];
	// Copy the meaningful part from compressedGLCM
	memcpy(metaGLCM, codifiedMatrix, metaGlcmLength * sizeof(int));
	free(codifiedMatrix);

	cout << endl << "Final MetaGLCM";
	printArray(metaGLCM,metaGlcmLength);
	// from now on metaGLCM[metaGlcmLength]

	printMetaGlcm(metaGLCM, metaGlcmLength, numberOfPairs, imgData.grayLevel);


	double features[16];
    computeFeatures(features,metaGLCM,metaGlcmLength,numberOfPairs, imgData.grayLevel);
	printFeatures(features);

	return 0;
}


