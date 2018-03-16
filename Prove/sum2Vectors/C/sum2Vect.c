#include <stdio.h>
#include <stdlib.h>
#define SIZE	(1<<29) // 2^31 si the maximum safe value

void VectorAdd(float *a, float *b, float *c, int n)
{
	int i;

	for (i=0; i < n; ++i)
		c[i] = a[i] + b[i];
}

int main()
{
	float *a, *b, *c;
	
	// Memory space for 3 arrays
	a = (float *)malloc(SIZE * sizeof(float));
	b = (float *)malloc(SIZE * sizeof(float));
	c = (float *)malloc(SIZE * sizeof(float));
	
	// Initialize
	for (int i = 0; i < SIZE; ++i)
	{
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}
	
	// Perform the operation
	VectorAdd(a, b, c, SIZE);

	// Commented out because printing on the cmd is very slow
	/* Print results
	for (int i = 0; i < SIZE; ++i)
		printf("c[%d] = %d\n", i, c[i]);

	*/


	// Clean the memory
	free(a);
	free(b);
	free(c);

	return 0;
}