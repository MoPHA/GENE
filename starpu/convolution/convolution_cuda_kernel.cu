#include <starpu.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "convolution_core.h"

  template <typename T>
__global__ void convolution_kernel(
    const T *d_f, // Padded matrix
    const unsigned int paddedN, // N+r
    const unsigned int paddedM, // M+r
    const T *d_g, // kernel
    const int r,  // radius
    T *d_h,       // output
    const unsigned int N, // N
    const unsigned int M  // M
    ) 
{
  // Set the padding size and filter size
  unsigned int paddingSize = r;
  unsigned int filterSize = 2 * r + 1;

  // Set the pixel coordinate. 
  // Threads in the padding size wont do anything.

  const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x + paddingSize;
  const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y + paddingSize;

  // The multiply-add operation for the pixel coordinate ( j, i )
  if( j >= paddingSize && j < paddedN - paddingSize && i >= paddingSize && i < paddedM - paddingSize )
  {
    unsigned int oPixelPos = ( i - paddingSize ) * N + ( j - paddingSize );

    d_h[oPixelPos] = 0.0;

    for( int k = -r; k <=r; k++ )
    {
      for( int l = -r; l <= r; l++ )
      {
        unsigned int iPixelPos = ( i + k ) * paddedN + ( j + l );
        unsigned int coefPos = ( k + r ) * filterSize + ( l + r );
        d_h[oPixelPos] += d_f[iPixelPos] * d_g[coefPos];
      }
    }

  }
}

inline unsigned int iDivUp( const unsigned int &a, const unsigned int &b )
{
  return ( a%b != 0 ) ? (a/b+1):(a/b);
}

#if 0
extern "C++" void compute_convolution_gpu(
    vector< TestFunction<float>* > &subi, 
    TestKernel<float> &g,
    vector< TestFunction<float>* > &subo, 
    int M, int N
    )
{
  int r = g.radius;
  printf("[GPU]: Compute convolution ... \n");

  // Allocate the memory on a device (corresponding to a smaller conv_matrix)
  // ---------------------------------------------------------------------------- 
  float *d_f = NULL;
  unsigned int paddedMatrixSizeByte = subi[0]->get_mem_size();
  cudaMalloc( reinterpret_cast<void **>(&d_f), paddedMatrixSizeByte );

  float *d_h = NULL; 
  unsigned int imageSizeByte = subo[0]->get_mem_size();
  cudaMalloc( reinterpret_cast<void **>(&d_h), imageSizeByte );

  float *d_g = NULL;
  unsigned int filterKernelSizeByte = g.get_mem_size();
  cudaMalloc( reinterpret_cast<void **>(&d_g), filterKernelSizeByte );

  float *h_g = g.data;        // Kernel
  cudaMemcpy( d_g, h_g, filterKernelSizeByte, cudaMemcpyHostToDevice ); // Host to Device

  // Setting the execution configuration
  // ---------------------------------------------------------------------------- 
  const unsigned int blockN = 32;
  const unsigned int blockM = 32;
  const dim3 grid( iDivUp( N, blockN ), iDivUp( M, blockM ) );
  const dim3 threadBlock( blockN, blockM );

  printf("Convolution GPU tasks ...\n");
  for (int i=0; i<subi.size(); i++)
  {
    float *h_f = subi[i]->data; // Input
    float *h_h = subo[i]->data; // Output

    // Transfer  from a host to a device
    cudaMemcpy(d_f, h_f, paddedMatrixSizeByte, cudaMemcpyHostToDevice ); // Host to Device

    // Convolve: call cuda kernel
    convolution_kernel<<<grid,threadBlock>>>(
        d_f, subi[i]->x_num, subi[i]->y_num, 
        d_g, r, 
        d_h, subo[i]->x_num, subo[i]->y_num); 

    cudaDeviceSynchronize();

    // Transfer result from the device to the host
    cudaMemcpy( h_h, d_h, imageSizeByte, cudaMemcpyDeviceToHost ); // Device to Host
  }
}
#endif

extern "C++" void compute_convolution_gpu_func(void *buffers[], void *cl_arg)
{
  float *fo, *fi, *fk;
  size_t no, mo, ni, mi, nk;

  int M, N;
  starpu_codelet_unpack_args(cl_arg, &M, &N);

  // These are cuda pointers
  fo = (float*)STARPU_MATRIX_GET_PTR(buffers[0]);
  no = (unsigned)STARPU_MATRIX_GET_NX(buffers[0]);
  mo = (unsigned)STARPU_MATRIX_GET_NY(buffers[0]);

  fi = (float*)STARPU_MATRIX_GET_PTR(buffers[1]);
  ni = (unsigned)STARPU_MATRIX_GET_NX(buffers[1]);
  mi = (unsigned)STARPU_MATRIX_GET_NY(buffers[1]);

  fk = (float*)STARPU_MATRIX_GET_PTR(buffers[2]);
  nk = (unsigned)STARPU_MATRIX_GET_NX(buffers[2]);

  int r = (nk-1)/2;
  /* printf("%d, %d,%d, %d, %d, %d, %d, %d\n", M, N, mi, ni,  mo, no, nk, r) ; */

#if 0
  const unsigned int blockN = 32;
  const unsigned int blockM = 32;
  const dim3 grid( iDivUp( N, blockN ), iDivUp( M, blockM ) );
  const dim3 threadBlock( blockN, blockM );

  // Convolve: call cuda kernel
  convolution_kernel<<<grid,threadBlock, starpu_cuda_get_local_stream()>>>(fi, ni, mi, fk, r, fo, no, mo);
  cudaDeviceSynchronize();
#endif

  const unsigned int blockN = 32;
  const unsigned int blockM = 32;
  const dim3 grid( iDivUp( N, blockN ), iDivUp( M, blockM ) );
  const dim3 threadBlock( blockN, blockM );
  // No shared memory: third parameter is 0
  convolution_kernel<<< grid, threadBlock, 0, starpu_cuda_get_local_stream()>>>(fi, ni, mi, fk, r, fo, no, mo);
  cudaStreamSynchronize(starpu_cuda_get_local_stream());
}


