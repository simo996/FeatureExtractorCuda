#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // Command Options
// File c++ libraries
#include <fstream>
#include <iostream>
#include <iterator>
#include <algorithm> 
// OpenCv Libraries for loading MRImages
#include <opencv2/opencv.hpp>
#include "GLCM.h"

using namespace cv; // Loading MRI images
using namespace std;

struct ImageData{
	int rows;
	int columns;
	int grayLevel; // 16_U, 16_S
	int distance;
};

/* Support Code */

void printArray(int * vector, int length)
{
	cout << endl;
	for (int i = 0; i < length; i++)
	{
		cout << vector[i] << " ";
	}
	cout << endl;
}

void sort(int * vector, int length) // Will modify the input vector
{
	int swap;
	for (int i = 0; i < length; i++)
	{
		for (int j = i; j < length; j++)
		{
			if(vector[i] > vector[j])
			{
				swap = vector[i];
				vector[i] = vector[j];
				vector[j] = swap;
			}
		}
	}
}

// Return the length of the compressed metaglcm
int compress(int * inputArray, int * outputArray, int length)
{
	int occurrences = 0;
	int deletions = 0;
	int j = 1;

	for (int i = 0; i < length; i++)
	{
		occurrences = 0;
		j = i+1;
		// Count multiple occurrences of the same number
		while((inputArray[i] != -1) && (inputArray[i] == inputArray [j]))
		{
			occurrences++;
			deletions++;
			inputArray[j] = -1; // destroy from collection
			j++;
		}
		// Increment quantity
		inputArray[i]=inputArray[i]+occurrences;

	}

	j = 0; // in the end equals to Length-deletions
	// Copy non -1 numbers in the output vector
	for (int i = 0; i < length; i++)
	{
		if(inputArray[i] != -1)
		{
			outputArray[j] = inputArray[i];
			j++;
		}
	}
	return j;
}

/* Program Routines */

void initialControls(int argc, char const *argv[])
{
	if (argc != 2)
	{
		fprintf(stderr, "Usage: FeatureExtractor imageFile\n");
		exit(-1);
	}
}


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
		imgData.distance = 1; // Default 
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


int main(int argc, char const *argv[])
{
	Mat imageMatrix; // Matrix representation of the image
	ImageData imgData; // MetaData about the image

	//initialControls(argc,argv);  TODO Analyze given options

	/* read image and extract metadata
	readMRImage(imageMatrix,imgData, argv[1]);

	*/

	// Mockup Matrix
	int testData[4][4] = {{0,0,1,1},{1,0,1,1},{0,2,2,2},{2,2,3,3}};
	imgData.rows = 4;
	imgData.columns = 4;
	imgData.grayLevel = 4;
	imgData.distance = 1;
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


	// Start Creating the first GLCM
	// 4x4 0° 1 pixel distanza

	int numberOfPairs = (imgData.columns-1) * (imgData.rows);
	assert(numberOfPairs==12);

	int * codifiedMatrix=(int *) malloc(sizeof(int)*numberOfPairs);
	int k=0;
	int referenceGrayLevel;
	int neighborGrayLevel;

	// FIRST STEP: codify all pairs
	for (int i = 0; i < imgData.rows; i++)
	{
		for (int j = 0; j < imgData.columns-1; j++)
		{
			referenceGrayLevel = imageMatrix.at<int>(i,j);
			neighborGrayLevel = imageMatrix.at<int>(i,j+1);

			codifiedMatrix[k] = ((referenceGrayLevel*imgData.grayLevel) + 
			neighborGrayLevel) * (numberOfPairs+1); // +1 teoricamente non serve
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
	int * compressedGLCM = (int *) malloc (sizeof(int)*numberOfPairs); // some dimension in excess
	int metaGlcmLength = compress(codifiedMatrix, compressedGLCM, numberOfPairs);

	int metaGLCM[metaGlcmLength];
	memcpy(metaGLCM, compressedGLCM, metaGlcmLength * sizeof(int));
	// Copy the meaningful part from compressedGLCM
	cout << endl << "Final MetaGLCM";
	printArray(metaGLCM,metaGlcmLength);
	// from now on metaGLCM[metaGlcmLength]
	return 0;
}