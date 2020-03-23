#ifndef _CONVOLUTION_UTILITIES_H
#define _CONVOLUTION_UTILITIES_H

#include <cstring>

const int NW=0;
const int NO=1;
const int NE=2;
const int WE=3;
const int AA=4;
const int EA=5;
const int SW=6;
const int SO=7;
const int SE=8;

const int DIMS = 2;
const int Y = 0;
const int X = 1;
const int n_neighbors = 9;
const int n_blocks = 9;

enum InitType
{
  ZERO=0,  // Zero
  INVA=1,  // -1
  CONS=2,  // 1:MN
  RAND=3,  // Pseudo Random
  NONE=4,  // No 
};

//============================================================================ 
// Utilities
//============================================================================

// Prints to the provided buffer a nice number of bytes (KB, mB, GB, etc)
static void pretty_bytes(char* buf, double  size)
{
  size_t i = 0;
  const char* units[] = {"B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"};
  while (size > 1024) {
    size /= 1024;
    i++;
  }
  sprintf(buf, "%4.3f_%s", size, units[i]);
}

  template<typename T>
void print_matrix(size_t m, size_t n, size_t l, T *data)
{
  printf("\n");
  for (size_t i=0; i<m; i++)
  {
    for (size_t j=0; j<n; j++)
    {
      printf("%5.2f ", (double)data[i*l+j]);
    }
    printf("\n");
  }
}

  template<typename T>
T compute_error(T* ref, T* fi, size_t m, size_t n)
{
  T error=0;
  for (size_t i=0; i<m; i++)
  {
    for (size_t j=0; j<n; j++)
    {
      size_t idx = i*n+j;
      error += abs(fi[idx]-ref[idx]);

      // if(error > 0)
      // {
      //   printf("%d %d %f %f\n", i, j, (double)fun[idx], (double)ref[idx]);
      //   return (T)0;
      // }
    }
  }
  return error;
}

  template<typename T>
void my_swap(T &p1, T &p2)
{
  T tmp; tmp = p1; p1 = p2; p2 = tmp;
}


  template <typename T>
void initialize_matrix(T* data, size_t m, size_t n, size_t l, InitType option)
{
  switch(option)
  {
    case ZERO:
      for (size_t i=0; i<m; i++)
      {
        for (size_t j=0; j<n; j++)
        {
          size_t idx = i*l+j;
          data[idx] = 0;
        }
      }
      break;
    case INVA:
      for (size_t i=0; i<m; i++)
      {
        for (size_t j=0; j<n; j++)
        {
          size_t idx = i*l+j;
          data[idx] = -1;
        }
      }
      break;
    case CONS:
      for (size_t i=0; i<m; i++)
      {
        for (size_t j=0; j<n; j++)
        {
          data[i*l+j] = i*n+j;
        }
      }
      break;

    case RAND:
      srand(1);
      for (size_t i=0; i<m; i++)
      {
        for (size_t j=0; j<n; j++)
        {
          size_t idx = i*l+j;
          data[idx] =  (T)(rand() /(double(RAND_MAX)));
        }
      }
      break;
    default:
      // Do nothing
      break;
  }
}

#endif
