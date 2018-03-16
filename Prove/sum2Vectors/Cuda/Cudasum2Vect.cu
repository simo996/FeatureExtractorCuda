#include <stdio.h>

#define SIZE	(1<<25) // Number of elements in the arrays
#define NTHREADS 1024 // Number of threads per block


__global__ void VectorAdd(int *a, int *b, int *c, int n)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	int shift= 1<<30/SIZE;
	int i;
	if (index < n)
		for(i=0;i<shift;i++)
		{
			c[index+i] = a[index+i] + b[index+i];
		}
}

int main()
{
	int *a, *b, *c;

	// Had issues with CudaMallocManaged
	int *d_a, *d_b, *d_c;

	// Memory space for 3 arrays on CPU

	a = (int *)malloc(SIZE*sizeof(int));
	b = (int *)malloc(SIZE*sizeof(int));
	c = (int *)malloc(SIZE*sizeof(int));

	// Memory space for 3 arrays on GPU

	cudaMalloc( &d_a, SIZE*sizeof(int));
	cudaMalloc( &d_b, SIZE*sizeof(int));
	cudaMalloc( &d_c, SIZE*sizeof(int));

	// Initialization
	for( int i = 0; i < SIZE; ++i )
	{
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}

	// Move Actual Data to GPU memory
	cudaMemcpy( d_a, a, SIZE*sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( d_b, b, SIZE*sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( d_c, c, SIZE*sizeof(int), cudaMemcpyHostToDevice );

	// Launche Kernels
	int nBlocks= (SIZE)/NTHREADS;
	if(nBlocks<(2<<16))
		VectorAdd<<< nBlocks, NTHREADS >>>(d_a, d_b, d_c, SIZE);
	else
		printf("Too many blocks for the architecture\n");
	
	cudaMemcpy( c, d_c, SIZE*sizeof(int), cudaMemcpyDeviceToHost );

	// Print some results
	/*Commented out because they slow the process
	for( int i = 0; i < SIZE; ++i)
		printf("c[%d] = %d\n", i, c[i]);
	*/

	// Clean all memory
	free(a);
	free(b);
	free(c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}