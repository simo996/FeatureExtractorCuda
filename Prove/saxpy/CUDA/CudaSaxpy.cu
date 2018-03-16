#include <stdio.h>
#include <stdlib.h>
#define SIZE (1<<25) // Maximum number of elements of the arrays
#define NTHREADS 1024// Maximum number of threads per block

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) // avoid issues when there are more threads than elements
    y[i] = a*x[i] + y[i];
}

int main(void)
{

  // Declare and allocate vectors on both host and device
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(SIZE*sizeof(float));
  y = (float*)malloc(SIZE*sizeof(float));

  cudaMalloc(&d_x, SIZE*sizeof(float)); 
  cudaMalloc(&d_y, SIZE*sizeof(float));

  // initialization on host
  for (int i = 0; i < SIZE; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // data transfer
  cudaMemcpy(d_x, x, SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, SIZE*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on SIZE elements
  int nBlocks=(SIZE+255)/NTHREADS;
  if (nBlocks > (2<<16))
    printf("too many blocks\n");
  else
  {
    saxpy<<<nBlocks, NTHREADS>>>(SIZE, 2.0f, d_x, d_y);

    // Read the results
    cudaMemcpy(y, d_y, SIZE*sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i = 0; i < SIZE; i++)
      maxError = max(maxError, abs(y[i]-4.0f));
    printf("Max error: %f\n", maxError);
  }


  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}