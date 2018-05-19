//
// Created by simo on 07/05/18.
//

#include "SupportCode.h"
#include <math.h>
#include <iostream>
#include <assert.h>
#include <climits>

using namespace std;

void printArray(const int * vector, const int length)
{
	cout << endl;
	for (int i = 0; i < length; i++)
	{
		cout << vector[i] << " ";
	}
	cout << endl;
}

/* Simple sort of an array */
void sort(int * vector, int length) // Will modify the input vector
{
	int swap;
	for (int i = 0; i < length; i++) {
		for (int j = i; j < length; j++) {
			if (vector[i] > vector[j]) {
				swap = vector[i];
				vector[i] = vector[j];
				vector[j] = swap;
			}
		}
	}
}

/* Will count how many different elements can be found into a sorted array
 * Require an ordered array */
int findUnique(int * inputArray, const int length)
{
	int uniqueElements = 0;
	int i = 0,j;

	while(i < length)
	{
		uniqueElements++;
		j=i+1;
		while(inputArray[i]==inputArray[j])
		{
			j++;
		}
		i=j;
	}
	return uniqueElements;
}

/*  Increment multiplicity of identical elements
	Redundant input items will be substituted with MAXINT
	Return the number of unique pairs that will occupy initial positions
	REQUIRES: a sorted array
*/
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

/*
	Return the (i,j)th element of a linearized matrix
*/
int getElementFromLinearMatrix(const int * input, const int nRows, const int nColumns, const int i, const int j)
{
	assert ((i <= nRows) && (j <= nColumns));
	return input[(i * nColumns) + j];
}

/*
void linearizeMatrix(const int nRows, const int nColumns, const int  input[][nColumns], int * output)
{
	for (int i = 0; i < nRows; ++i)
	{
		for(int j = 0 ; j < nColumns; j++)
		{
			output [i * nColumns + j] = input[i][j];
		}
	}
}
 */

/*
	Copy into output array just unique elements
	Increment multiplicity of identical elements
	Redundant input items will be substituted with -1
*/
void compress(int * inputArray, int * outputArray, const int length)
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
		inputArray[i] = inputArray[i] + occurrences;
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
}
