//
// Created by simo on 07/05/18.
//

#include "SupportCode.h"
#include <math.h>
#include <iostream>

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

