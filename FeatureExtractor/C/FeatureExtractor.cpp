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

struct GrayPair{
	int grayLevelI;
	int grayLevelJ;
	int multiplicity;
};

struct AggregatedPair{
	int aggregatedGrayLevels; //i+j or abs(i-j)
	int multiplicity;
};


// Extract from a pair of gray levels i,g and their multiplicity
struct GrayPair unPack(const int value, const int numberOfPairs, const int maxGrayLevel)
{
  struct GrayPair couple;
  int roundDivision = value / numberOfPairs; // risultato intero
  couple.multiplicity = (value - roundDivision * numberOfPairs)+1;
  couple.grayLevelI = roundDivision / maxGrayLevel; // risultato intero
  couple.grayLevelJ = roundDivision - (maxGrayLevel * couple.grayLevelI);
  return couple;
};

struct AggregatedPair aggregatedPairUnPack(const int value, const int numberOfPairs)
{
	struct AggregatedPair pair;
	int roundDivision = value / numberOfPairs;
	pair.aggregatedGrayLevels = roundDivision; // risultato intero
	pair.multiplicity = value - roundDivision * numberOfPairs ;
	return pair;
};


/* Support Code */

void printArray(const int * vector, const int length)
{
	cout << endl;
	for (int i = 0; i < length; i++)
	{
		cout << vector[i] << " ";
	}
	cout << endl;
}

void printMetaGlcm(const int * metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel){
	GrayPair actual;
	for (int i = 0; i < length; ++i)
	{
		actual = unPack(metaGLCM[i], numberOfPairs, maxGrayLevel);
		cout << endl << "i: " << actual.grayLevelI 
			<< "\tj: " << actual.grayLevelJ 
			<< "\tmult: " << actual.multiplicity ;
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

/* GLCM Code */
void initializeGLCM(struct GLCMData glcm0, int distance, int shiftX, int shiftY)
{
	
}

// Add identical elements into a new element
// Return the length of the compressed metaglcm
int compress(int * inputArray, int * outputArray, const int length)
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

// Same as compress but changing the InputArray and returning the final length
int localCompress(int * inputArray, const int length)
{
	int occurrences = 0;
	int deletions = 0;
	int j = 1;

	for (int i = 0; i < length; i++)
	{
		occurrences = 0;
		j = i+1;
		// Count multiple occurrences of the same number
		while((inputArray[i] != INT_MAX) && (inputArray[i] == inputArray [j]))
		{
			occurrences++;
			deletions++;
			inputArray[j] = INT_MAX; // for destroying
			j++;
		}
		// Increment quantity
		if(inputArray[j] != INT_MAX){
			inputArray[i]=inputArray[i]+occurrences;
		}

	}

	sort(inputArray,length);
	// After |length| numbers there should be only INT_MAX
	return length-deletions;
}

// Same pair with different multeplicty collapsed into a single element
int compressMultiplicity(int * inputArray, const int length, const int numberOfPairs, const int imgGrayLevel){
	int occurrences = 0;
	int deletions = 0;
	int j = 1;

	for (int i = 0; i < length; i++)
	{
		occurrences = 0;
		j = i+1;
		// Count multiple occurrences of the same number
				//DIVISIONE PER ZERO, PER GIOVE, CRASHA
		while((inputArray[i] != INT_MAX) && (inputArray[i] % inputArray [j]) < numberOfPairs)
		{
			occurrences+=unPack(inputArray[j],numberOfPairs,imgGrayLevel).multiplicity;
			deletions++;
			inputArray[j] = INT_MAX; // for destroying
			j++;
		}
		// Increment quantity
		if(occurrences>=numberOfPairs){
			fprintf(stderr, "Error while compressing minding the multiplicity\n");
			fprintf(stderr, "Summed Multiplicity: %d\n",occurrences);
			fprintf(stderr, "Element: %d\n",inputArray[i]);

			exit(-1);
		}
		if(inputArray[j] != INT_MAX){
			inputArray[i] = inputArray[i]+occurrences;
		}

	}

	sort(inputArray,length);
	// After |length| numbers there should be only INT_MAX
	return length-deletions;

}

// Adapt the GLCM to include ulterior elements already codified
// Will return the modified array and its length
int addElements(int * metaGLCM, int * elementsToAdd, int * outputArray, const int initialLength, const int numElements, const int numberOfPairs, const int grayLevel)
{

	// Combine the 2 arrays
	int sumLength= numElements + initialLength;

	int i,j;
	for ( i = 0; i < initialLength; ++i) {
		outputArray[i] = metaGLCM[i];
	}
	for(j=0; j< numElements; j++)
	{
		outputArray[i]= elementsToAdd[j];
		i++;
	}
	printArray(outputArray,sumLength);

	// Sort and compress identical pairs
	sort(outputArray, sumLength); // Required for successive compresses
	int reducedLength = localCompress(outputArray, sumLength);
	printArray(outputArray,reducedLength);


	// Same pair with different molteplicity will be compressed

	// PERCHÈ NON RITORNA???
	int finalReducedLength = compressMultiplicity(outputArray,reducedLength,numberOfPairs,grayLevel);
	printArray(outputArray,finalReducedLength);
	return finalReducedLength;
}


// Reduce the molteplicity into the metaGLCM of elements given
// ASSUMPTION all given elements can be found in metaGLCM
void dwarf(int * metaGLCM, int * listElements, int lengthGlcm, int lengthElements)
{
	// both arrays need to be ordered
	sort(listElements, lengthElements);
	int j = 0;
	
	for (int i = 0; i < lengthElements; ++i)
	{
		while(metaGLCM[j] != listElements [i]){
			j++;
		}
		//int multiplicity = listElements[i].unPack(...).multiplicity;
		// metaGLCM[j]-=multiplicity;
	}
}



// FEATURES

double computeASM(const int * metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel)
{
	double angularSecondMoment = 0;
	GrayPair actualPair;
	double actualPairProbability;    
	
	for(int i=0 ; i<length; i++)
    {
        actualPair = unPack(metaGLCM[i], numberOfPairs, maxGrayLevel);
        actualPairProbability = actualPair.multiplicity/numberOfPairs;
        
        angularSecondMoment += pow((actualPairProbability),2);
    }
    return angularSecondMoment;

}

double computeAutocorrelation(const int * metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel)
{
	double autocorrelation = 0;
	GrayPair actualPair;
	double actualPairProbability;
    
    for(int i=0 ; i<length; i++)
    {
        actualPair = unPack(metaGLCM[i], numberOfPairs, maxGrayLevel);
        actualPairProbability = actualPair.multiplicity/numberOfPairs;
        
        autocorrelation += actualPair.grayLevelI * actualPair. grayLevelJ * actualPairProbability;
    }
    return autocorrelation;
}


double computeEntropy(const int * metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel)
{
	double entropy = 1;
	GrayPair actualPair;
	double actualPairProbability;    
	for(int i=0 ; i<length; i++)
    {
        actualPair = unPack(metaGLCM[i], numberOfPairs, maxGrayLevel);
        actualPairProbability = actualPair.multiplicity/numberOfPairs;
        
        entropy += actualPairProbability * log(actualPairProbability); 
        // No pairs with 0 probability, so log is safe
    }
    return (-1*entropy);
}

double computeMaximumProbability(const int * metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel)
{
	double maxProb;
	GrayPair actualPair = unPack(metaGLCM[0], numberOfPairs, maxGrayLevel);
	double actualPairProbability = actualPair.multiplicity/numberOfPairs;
	maxProb = actualPairProbability;

    for(int i=1 ; i<length; i++)
    {
        actualPair = unPack(metaGLCM[i], numberOfPairs, maxGrayLevel);
        actualPairProbability = actualPair.multiplicity/numberOfPairs;

        if(actualPairProbability > maxProb)
        {
        	maxProb = actualPairProbability;
        }
    }
    return maxProb;
}

double computeHomogeneity(const int * metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel)
{
	double homogeneity = 0;
	GrayPair actualPair;
	double actualPairProbability;

    for(int i=0 ; i<length; i++)
    {
        actualPair = unPack(metaGLCM[i], numberOfPairs, maxGrayLevel);
        actualPairProbability = actualPair.multiplicity/numberOfPairs;
        
        homogeneity += actualPairProbability /
         (1 + fabs(actualPair.grayLevelI - actualPair.grayLevelJ));
        
    }
    return homogeneity;
}


double computeContrast(const int * metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel)
{
	double contrast = 0;
	GrayPair actualPair;
	double actualPairProbability;

    for(int i=0 ; i<length; i++)
    {
        actualPair = unPack(metaGLCM[i], numberOfPairs, maxGrayLevel);
        actualPairProbability = actualPair.multiplicity/numberOfPairs;
        
        contrast += actualPairProbability 
        * (pow(fabs(actualPair.grayLevelI - actualPair.grayLevelJ), 2));
        
    }
    return contrast;
}

double computeCorrelation(const int * metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel, const double muX, const double muY, const double sigmaX, const double sigmaY)
{
	double correlation = 0;
	GrayPair actualPair;
	double actualPairProbability;

    for(int i=0 ; i<length; i++)
    {
        actualPair = unPack(metaGLCM[i], numberOfPairs, maxGrayLevel);
        actualPairProbability = actualPair.multiplicity/numberOfPairs;
        
        correlation += ((actualPair.grayLevelI - muX) * (actualPair.grayLevelJ - muY) * actualPairProbability )
        /(sigmaX * sigmaY);
        
    }
    return correlation;
}

double computeClusterProminecence(const int * metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel, const double muX, const double muY)
{
	double clusterProminecence = 0;
	GrayPair actualPair;
	double actualPairProbability;

    for(int i=0 ; i<length; i++)
    {
        actualPair = unPack(metaGLCM[i], numberOfPairs, maxGrayLevel);
        actualPairProbability = actualPair.multiplicity/numberOfPairs;
        
        clusterProminecence += pow ((actualPair.grayLevelI + actualPair.grayLevelJ -muX - muY), 4) * actualPairProbability;
    }
    return clusterProminecence;
}

double computeClusterShade(const int * metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel, const double muX, const double muY)
{
	double clusterShade = 0;
	GrayPair actualPair;
	double actualPairProbability;

    for(int i=0 ; i<length; i++)
    {
        actualPair = unPack(metaGLCM[i], numberOfPairs, maxGrayLevel);
        actualPairProbability = actualPair.multiplicity/numberOfPairs;
        
        clusterShade += pow ((actualPair.grayLevelI + actualPair.grayLevelJ -muX - muY), 3) * actualPairProbability;
    }
    return clusterShade;
}

double computeSumOfSquares(const int * metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel, const double mu)
{
	double sumSquares = 0;
	GrayPair actualPair;
	double actualPairProbability;

    for(int i=0 ; i<length; i++)
    {
        actualPair = unPack(metaGLCM[i], numberOfPairs, maxGrayLevel);
        actualPairProbability = actualPair.multiplicity/numberOfPairs;
        
        sumSquares += pow ((actualPair.grayLevelI - mu), 2) * actualPairProbability;
    }
    return sumSquares;
}

double computeInverceDifferentMomentNormalized(const int * metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel)
{
	double inverceDifference = 0;
	GrayPair actualPair;
	double actualPairProbability;

    for(int i=0 ; i<length; i++)
    {
        actualPair = unPack(metaGLCM[i], numberOfPairs, maxGrayLevel);
        actualPairProbability = actualPair.multiplicity/numberOfPairs;
        
        inverceDifference += actualPairProbability / 
        (((1+pow(actualPair.grayLevelI - actualPair.grayLevelJ),2))/maxGrayLevel);
    }
    return inverceDifference;
}

// SUM Aggregated features
double computeSumAverage(const int * summedMetaGLCM, const int length, const int numberOfPairs)
{
	double result = 0;
	AggregatedPair actualPair;
	double actualPairProbability;

    for(int i=0 ; i<length; i++)
    {
        actualPair = aggregatedPairUnPack(summedMetaGLCM[i], numberOfPairs);
        actualPairProbability = actualPair.multiplicity/numberOfPairs;
        
        result += actualPair.aggregatedGrayLevels * actualPairProbability;
    }
    return result;
}

double computeSumEntropy(const int * summedMetaGLCM, const int length, const int numberOfPairs)
{
	double result = 0;
	AggregatedPair actualPair;
	double actualPairProbability;

    for(int i=0 ; i<length; i++)
    {
        actualPair = aggregatedPairUnPack(summedMetaGLCM[i], numberOfPairs);
        actualPairProbability = actualPair.multiplicity/numberOfPairs;
        
        result += log(actualPair.aggregatedGrayLevels) * actualPairProbability;
    }
    return result;
}

double computeSumVariance(const int * summedMetaGLCM, const int length, const int numberOfPairs, const double sumEntropy)
{
	double result = 0;
	AggregatedPair actualPair;
	double actualPairProbability;

    for(int i=0 ; i<length; i++)
    {
        actualPair = aggregatedPairUnPack(summedMetaGLCM[i], numberOfPairs);
        actualPairProbability = actualPair.multiplicity/numberOfPairs;
        
        result += ((actualPair.aggregatedGrayLevels - sumEntropy)^2) 
        * actualPairProbability;
    }
    return result;
}

double computeDifferenceEntropy(const int * aggregatedMetaGLCM, const int length, const int numberOfPairs)
{
	double result = 0;
	AggregatedPair actualPair;
	double actualPairProbability;

    for(int i=0 ; i<length; i++)
    {
        actualPair = aggregatedPairUnPack(aggregatedMetaGLCM[i], numberOfPairs);
        actualPairProbability = actualPair.multiplicity/numberOfPairs;
        
        result += log(actualPairProbability) * actualPairProbability;
    }
    return -1*result;
}

double computeDifferenceVariance(const int * aggregatedMetaGLCM, const int length, const int numberOfPairs)
{
	double result = 0;
	AggregatedPair actualPair;
	double actualPairProbability;

    for(int i=0 ; i<length; i++)
    {
        actualPair = aggregatedPairUnPack(aggregatedMetaGLCM[i], numberOfPairs);
        actualPairProbability = actualPair.multiplicity/numberOfPairs;
        
        result += (actualPair.aggregatedGrayLevels^2) * actualPairProbability;
    }
    return result;
}


// Mean of all probabilities
double computeMean(const int* metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel)
{	
	double mu = 0;
  	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i<length; i++)
    {
        actualPair = unPack(metaGLCM[i], numberOfPairs, maxGrayLevel);
        actualPairProbability = actualPair.multiplicity/numberOfPairs;
    	
    	muX += actualPairProbability; 
 	}
 	return mu;
}

// Mean of
double computeMuX(const int* metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel)
{	
	double muX = 0;
  	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i<length; i++)
    {
        actualPair = unPack(metaGLCM[i], numberOfPairs, maxGrayLevel);
        actualPairProbability = actualPair.multiplicity/numberOfPairs;
    	
    	muX += actualPair.grayLevelI * actualPairProbability; 
 	}
 	return muX;
}

double computeMuY(const int* metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel)
{	
	double muY = 0;
  	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i<length; i++)
    {
        actualPair = unPack(metaGLCM[i], numberOfPairs, maxGrayLevel);
        actualPairProbability = actualPair.multiplicity/numberOfPairs;
    	
    	muY += actualPair.grayLevelJ * actualPairProbability; 
 	}
 	return muY;
}

double computeSigmaX(const int* metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel, const double muX)
{	
	double sigmaX = 0;
  	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i<length; i++)
	{
		actualPair = unPack(metaGLCM[i], numberOfPairs, maxGrayLevel);
		actualPairProbability = actualPair.multiplicity/numberOfPairs;

		sigmaX += ((actualPair.grayLevelI - muX)^2) * actualPairProbability; 
 	}

 	return sqrt(sigmaX);
}

double computeSigmaY(const int* metaGLCM, const int length, const int numberOfPairs, const int maxGrayLevel, const double muY)
{	
	double sigmaY = 0;
  	GrayPair actualPair;
	double actualPairProbability;

	for(int i=0 ; i<length; i++)
	{
		actualPair = unPack(metaGLCM[i], numberOfPairs, maxGrayLevel);
		actualPairProbability = actualPair.multiplicity/numberOfPairs;

		sigmaY += ((actualPair.grayLevelJ - muY)^2) * actualPairProbability; 
 	}

 	return sqrt(sigmaY);
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

	// TEST FOR ADDING CODIFIED ELEMENTS


	//DIVISIONE PER ZERO, PER GIOVE, CRASHA
	if(testAddElements(metaGLCM,metaGlcmLength, numberOfPairs, imgData.grayLevel))

	return 0;
}


