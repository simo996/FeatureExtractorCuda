#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // Command Options
// File c++ libraries
#include <fstream>
#include <iostream>
// OpenCv Libraries for loading MRImages
#include <opencv2/opencv.hpp>
#include "GLCM.h"

using namespace cv; // Loading MRI images
using namespace std;

struct ImageData{
	int rows;
	int columns;
	int grayLevel; // 16_U, 16_S
};

void readMRImage(Mat image, struct ImageData imgData, const char * filename)
{
	image = imread(filename, CV_LOAD_IMAGE_ANYDEPTH );
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


int main(int argc, char const *argv[])
{

	// TODO Analyze options

	Mat imageMatrix; // Matrix representation of the image
	ImageData imgData; // MetaData about the image

	// read image and extract metadata
	readMRImage(imageMatrix,imgData, argv[1]); 


	return 0;
}