//
// Created by simo on 07/05/18.
//

#ifndef FEATURESEXTRACTOR_SUPPORTCODE_H
#define FEATURESEXTRACTOR_SUPPORTCODE_H

void printArray(const int * vector, const int length);
void sort(int * vector, int length);
int findUnique(int * inputArray, const int length);
void compress(int * inputArray, int * outputArray, const int length);
int localCompress(int * inputArray, const int length);
int getElementFromLinearMatrix(const int * input, const int nRows, const int nColumns, const int i, const int j);

#endif //FEATURESEXTRACTOR_SUPPORTCODE_H
