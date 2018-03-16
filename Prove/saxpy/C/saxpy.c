#include <stdio.h>
#include <stdlib.h>

#define SIZE (1<<29) // Do not exceed device's available ram
#define VALUE 2

void saxpy(int n, float a, float * restrict x, float * restrict y)
{
  for (int i = 0; i < n; ++i)
      y[i] = a*x[i] + y[i];
}

int main()
{
	float * vectorX = (float *) malloc(SIZE * sizeof(float));
	float * vectorY = (float *) malloc(SIZE * sizeof(float));

	// Initialize vectorX with some data
	for(int i = 0; i < SIZE; i++)
	{
		vectorX[i] =i;
	}

	// Perform SAXPY on 1B elements
	saxpy(SIZE, VALUE, vectorX, vectorY);

	// print some results to see if correct
	/* 
	for(int i = 0; i < 100; i++)
	{
		printf("y[%d] = %f\n", i, vectorY[i]);
	}
	*/

	free(vectorX);
	free(vectorY);
}