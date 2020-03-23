#ifndef _CONVOLUTION_KERNEL_H
#define _CONVOLUTION_KERNEL_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <vector>
#include <string>

#include "convolution_utilities.h"

//============================================================================ 
// Kernel Definition
//============================================================================

using namespace std;

//Create kernel
template<typename T> 
class Kernel2D
{
  public:
    size_t r;
    size_t l;
    size_t m;
    size_t n;
    T* data;

    Kernel2D(size_t r): r(r), data(NULL)
  {
    m = (2*r + 1);
    n = m;
    l = n;
    data = (T*)malloc(sizeof(T)*m*n); 
  }
    ~Kernel2D()
    {
      if(data) {free(data); data=NULL; }
      n = 0;
      m = 0;
      l = 0;
    }

    size_t get_mem_size() const 
    { return (sizeof(T)*m*n); }
};

template<typename T>
class TestKernel : public Kernel2D<T> 
{
  public:

    TestKernel(size_t r, int option=-1) : Kernel2D<T>(r)
  {

    T* data = this->data;
    size_t m = this->m;
    size_t n = this->n;
    size_t l = this->l;

    for (size_t i=0; i<m; i++)
    {
      for (size_t j=0; j<n; j++)
      {
        size_t idx = i*l+j;
        if(option == -1) 
        {
          data[idx] = 1; //(T)(idx);
        }
        else if(option==0)
        {
          size_t off_i=r;
          size_t off_j=r;
          data[idx] = 0;
          data[off_i*n+ off_j] = 1;
        }
        else if(option==1)
        {
          data[idx] = option;
        }
        else
        {
          printf("Test Kernel, invalid initialization option\n");
        }
      }
    }
  }
};

template<typename T>
class meanKernel : public Kernel2D<T> 
{
  public:

    meanKernel(size_t r) : Kernel2D<T>(r)
  {

    T* data = this->data;
    size_t m = this->m;
    size_t n = this->n;
    size_t l = this->l;

    double factor = 1 /(m*n);
    for (size_t i=0; i<m; i++)
    {
      for (size_t j=0; j<n; j++)
      {
        size_t idx = i*l+j;
        data[idx] = factor;
      }
    }
  }
};

template<typename T>
class FiniteDiffOperator_x : public Kernel2D<T> 
{
  public:

    FiniteDiffOperator_x(double h) :Kernel2D<T>(1)
  {

    T* data = this->data;
    size_t m = this->m;
    size_t n = this->n;
    size_t l = this->l;

    double f = 1/(h*h);

    memset(data, m*l, 0);
    data[3] =  1.0*f;
    data[4] = -2.0*f;
    data[5] =  1.0*f;
  }
};

template<typename T>
class FiniteDiffOperator_y : public Kernel2D<T> 
{
  public:

    FiniteDiffOperator_y(double h) :Kernel2D<T>(1)
  {

    T* data = this->data;
    size_t m = this->m;
    size_t n = this->n;
    size_t l = this->l;

    double f = 1/(h*h);

    memset(data, m*l, 0);
    data[1] =  1.0*f;
    data[4] = -2.0*f;
    data[7] =  1.0*f;
  }
};

template<typename T>
class LaplaceOperator5: public Kernel2D<T> 
{
  public:
    LaplaceOperator5(double h) :Kernel2D<T>(1)
  {

    T* data = this->data;
    size_t m = this->m;
    size_t n = this->n;
    size_t l = this->l;

    double f = 1/(h*h);
    memset(data, m*l, 0);
    data[1] =  1.0*f;
    data[3] =  1.0*f;
    data[4] = -4.0*f;
    data[5] =  1.0*f;
    data[7] =  1.0*f;
  }
};

template<typename T>
class LaplaceOperator9: public Kernel2D<T> 
{
  public:
    LaplaceOperator9() :Kernel2D<T>(1)
  {

    T* data = this->data;
    size_t m = this->m;
    size_t n = this->n;
    size_t l = this->l;

    data[NW] =   1.0;
    data[NO] =   4.0;
    data[NE] =   1.0;

    data[WE] =   4.0;
    data[AA] = -20.0;
    data[EA] =   4.0;

    data[SW] =   1.0;
    data[SO] =   4.0;
    data[SE] =   1.0;
  }
};

#endif
