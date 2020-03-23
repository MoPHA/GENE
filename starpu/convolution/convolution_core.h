#ifndef _CONVOLUTION_CORE_H
#define _CONVOLUTION_CORE_H

#include "convolution_kernels.h"
#include <stdlib.h>
#include <limits>

using namespace std;

//============================================================================ 
// Convolution Definition
//============================================================================

template <typename T>
void load_ghost_cells( 
    T* datai, size_t &mi, size_t &ni, size_t &li, 
    T* datao, size_t &mo, size_t &no, size_t &lo, 
    int ibuf, size_t r)
{
  size_t off_i, off_j; //offsets
  size_t up_i, up_j; //upper limit in i
  size_t lo_i, lo_j; //lower limit in j

  lo_i=0; lo_j=0;

  switch(ibuf)
  {
    case NW:
      up_i=r;  off_i=0; 
      up_j=r;  off_j=0;
      break;
    case NO:
      up_i=r ; off_i=0;
      up_j=ni; off_j=0;
      break;
    case NE:
      up_i=r; off_i=0;  
      up_j=r; off_j=ni-r;  
      break;
    case WE:
      up_i=mi; off_i=0;
      up_j=r ; off_j=0;
      // printf("[%d] %ld %ld %ld %ld\n", ibuf, up_i, up_j,  off_i, off_j);
      break;
    case EA:
      up_i=mi; off_i=0;
      up_j=r ; off_j=ni-r;
      break;
    case SW:
      up_i=r;  off_i=mi-r; 
      up_j=r;  off_j=0;
      break;
    case SO:
      up_i=r ; off_i=mi-r;
      up_j=ni; off_j=0;
      break;
    case SE:
      up_i=r; off_i=mi-r;  
      up_j=r; off_j=ni-r;  
      break;
    default:
      return;
      // Do nothing
  }

  for (size_t i=lo_i; i<up_i; i++)
  {
    for (size_t j=lo_j; j<up_j; j++)
    {
      datao[i*lo+j] = datai[(off_i+i)*li+(off_j+j)]; 
    }
  }	
}

template<typename T> 
class Function2D
{
  public:
    size_t y,x;  // offset in rows and cols, respectively.
    size_t l;    // l
    size_t m;    // rows
    size_t n;    // cols
    T* data;
    vector<int> nb;            // neighbors
    vector<Function2D<T>*> gr; // ghost cells recv
    vector<Function2D<T>*> gs; // ghost cells send

    Function2D(size_t m, size_t n, size_t y=0, size_t x=0) 
    {
      this->m=m;
      this->n=n;
      this->y=y;
      this->x=x;
      l=n; data=NULL;

      if( m*n > std::numeric_limits<size_t>::max())
      {
        printf("WARNING(1): n*m > numeric_limits<size_t>::max()\n");
        printf("Indexing might be wrong \n");
      }


      if( sizeof(T)*m*n > std::numeric_limits<size_t>::max())
      {
        printf("WARNING(2): required memory larger than > numeric_limits<size_t>::max()\n");
      }

      data = (T*)malloc(sizeof(T)*m*n); 

      for (int i=0; i<n_neighbors; i++)
      {
        gr.push_back(NULL); 
        gs.push_back(NULL);
        nb.push_back(-1);
      }
    }

    size_t get_num() { return m*n; }
    size_t get_mem_size() {return (sizeof(T)*m*n);}

    ~Function2D()
    {
      if(data) { free(data); data=NULL; }
      for(int i=0; i<n_neighbors; i++)
      {
        if(gr[i]){delete gr[i]; gr[i]=NULL;}
        if(gs[i]){delete gs[i]; gs[i]=NULL;}
      }
      n = 0;
      m = 0;
    }

    void allocate_buffers_for_ghost_cells(const Kernel2D<T> &g)
    {
      bool pdebug=1;
      size_t r = g.r;

      // To recv
      if(nb[NW]!=-1)
      {
        // if(pdebug) printf("Allocate: %d\n", NW);
        gr[NW] = new Function2D<T>(r,r);
        gs[NW] = new Function2D<T>(r,r);
      }
      if(nb[NO]!=-1) 
      {
        // if(pdebug) printf("Allocate: %d\n", NO);
        gr[NO] = new Function2D<T>(r,n);
        gs[NO] = new Function2D<T>(r,n);
      }
      if(nb[NE]!=-1) 
      {
        // if(pdebug) printf("Allocate: %d\n", NE);
        gr[NE] = new Function2D<T>(r,r);
        gs[NE] = new Function2D<T>(r,r);
      }
      if(nb[WE]!=-1) 
      {
        // if(pdebug) printf("Allocate: %d\n", WE);
        gr[WE] = new Function2D<T>(m,r);
        gs[WE] = new Function2D<T>(m,r);
      }
      if(nb[EA]!=-1) 
      {
        // if(pdebug) printf("Allocate: %d\n", EA);
        gr[EA] = new Function2D<T>(m,r);
        gs[EA] = new Function2D<T>(m,r);
      }
      if(nb[SW]!=-1)
      {
        // if(pdebug) printf("Allocate: %d\n", SW);
        gr[SW] = new Function2D<T>(r,r);
        gs[SW] = new Function2D<T>(r,r);
      }
      if(nb[SO]!=-1)
      {
        // if(pdebug) printf("Allocate: %d\n", SO);
        gr[SO] = new Function2D<T>(r,n);
        gs[SO] = new Function2D<T>(r,n);
      }
      if(nb[SE]!=-1)
      {
        // if(pdebug) printf("Allocate: %d\n", SE);
        gr[SE] = new Function2D<T>(r,r);
        gs[SE] = new Function2D<T>(r,r);
      }
    }

    void load_ghost_cells_to_send(const Kernel2D<T> &g)
    {
      if(nb[NW]!=-1) load_ghost_cells(data, m, n, l, gs[NW]->data, gs[NW]->m, gs[NW]->n, gs[NW]->l, NW, g.r);
      if(nb[NO]!=-1) load_ghost_cells(data, m, n, l, gs[NO]->data, gs[NO]->m, gs[NO]->n, gs[NO]->l, NO, g.r);
      if(nb[NE]!=-1) load_ghost_cells(data, m, n, l, gs[NE]->data, gs[NE]->m, gs[NE]->n, gs[NE]->l, NE, g.r);
      if(nb[WE]!=-1) load_ghost_cells(data, m, n, l, gs[WE]->data, gs[WE]->m, gs[WE]->n, gs[WE]->l, WE, g.r);
      if(nb[EA]!=-1) load_ghost_cells(data, m, n, l, gs[EA]->data, gs[EA]->m, gs[EA]->n, gs[EA]->l, EA, g.r);
      if(nb[SW]!=-1) load_ghost_cells(data, m, n, l, gs[SW]->data, gs[SW]->m, gs[SW]->n, gs[SW]->l, SW, g.r);
      if(nb[SO]!=-1) load_ghost_cells(data, m, n, l, gs[SO]->data, gs[SO]->m, gs[SO]->n, gs[SO]->l, SO, g.r);
      if(nb[SE]!=-1) load_ghost_cells(data, m, n, l, gs[SE]->data, gs[SE]->m, gs[SE]->n, gs[SE]->l, SE, g.r);
    }

    void print_ghost_cells(int option, int my_rank)
    {
      vector<Function2D<T>*> &gc = (option==0)? gs : gr;
      for (int i=0; i<nb.size(); i++)
      {
        if(nb[i]==-1) continue;

        printf("[%d]: %d ----------------------------------------------------------------------------\n", my_rank, i);
        print_matrix( gc[i]->m,  gc[i]->n,  gc[i]->l,  gc[i]->data); 
      }
      printf("done\n");
    }

    void dump_data_to_file(string filename, int my_rank)
    {
      FILE *f = fopen(filename.c_str(), "a");
      if (f == NULL)
      {
        printf("Error opening file!\n");
        exit(1);
      }

      fprintf(f, "[%d]\n", my_rank);
      for (size_t i=0; i<m; i++)
      {
        for (size_t j=0; j<n; j++)
        {
          fprintf(f, "%5.0f ", data[i*l+j]);
        }
        fprintf(f, "\n");
      }
      fclose(f);
    }


};

template<typename T> 
class TestFunction : public Function2D<T>
{
  public:
    TestFunction(size_t m, size_t n, InitType inittype) : Function2D<T>(m, n)
  {
    T* data = this->data;
    size_t l = this->l;

    initialize_matrix(data, m, n, l, inittype);
  }
};

template<typename T> 
class TestFunctionConv : public Function2D<T>
{
  public:
    TestFunctionConv(size_t m_sub, size_t n_sub, size_t r, InitType inittype) : Function2D<T>(m_sub+2*r, n_sub+2*r)
  {
    size_t l = this->l;
    size_t m = this->m;
    size_t n = this->n;
    T* data_sub = &this->data[r*l+r];

    initialize_matrix(data_sub, m_sub, n_sub, l, inittype);

    T *data = this->data;
    T *datao = NULL; 
    size_t mo, no;
    datao = &data[0*l+0]        ; mo =r    ; no=r    ; load_ghost_cells(data_sub, m_sub, n_sub, l, datao, mo, no, l, SE, r) ;
    datao = &data[0*l+r]        ; mo =r    ; no=n_sub; load_ghost_cells(data_sub, m_sub, n_sub, l, datao, mo, no, l, SO, r) ;
    datao = &data[0*l+(n-r)]    ; mo =r    ; no=r    ; load_ghost_cells(data_sub, m_sub, n_sub, l, datao, mo, no, l, SW, r) ;
    datao = &data[r*l+0]        ; mo =m_sub; no=r    ; load_ghost_cells(data_sub, m_sub, n_sub, l, datao, mo, no, l, EA, r) ;
    datao = &data[r*l+(n-r)]    ; mo =m_sub; no=r    ; load_ghost_cells(data_sub, m_sub, n_sub, l, datao, mo, no, l, WE, r) ;
    datao = &data[(m-r)*l+0]    ; mo =r    ; no=r    ; load_ghost_cells(data_sub, m_sub, n_sub, l, datao, mo, no, l, NE, r) ;
    datao = &data[(m-r)*l+r]    ; mo =r    ; no=n_sub; load_ghost_cells(data_sub, m_sub, n_sub, l, datao, mo, no, l, NO, r) ;
    datao = &data[(m-r)*l+(n-r)]; mo =r    ; no=r    ; load_ghost_cells(data_sub, m_sub, n_sub, l, datao, mo, no, l, NW, r) ;

  }

    TestFunctionConv(TestFunction<T> &f, size_t r) : Function2D<T>(f.m+2*r, f.n+2*r)
  {
    size_t l = this->l;
    size_t m = this->m;
    size_t n = this->n;
    T* data_sub = &this->data[r*l+r];
    size_t m_sub = f.m;
    size_t n_sub = f.n;

    for(int i=0; i<f.m; i++) 
      for(int j=0; j<f.n; j++) 
        data_sub[i*l+j] = f.data[i*f.l+j];

    T *data = this->data;
    T *datao = NULL; 
    size_t mo, no;
    datao = &data[0*l+0]        ; mo =r    ; no=r    ; load_ghost_cells(data_sub, m_sub, n_sub, l, datao, mo, no, l, SE, r) ;
    datao = &data[0*l+r]        ; mo =r    ; no=n_sub; load_ghost_cells(data_sub, m_sub, n_sub, l, datao, mo, no, l, SO, r) ;
    datao = &data[0*l+(n-r)]    ; mo =r    ; no=r    ; load_ghost_cells(data_sub, m_sub, n_sub, l, datao, mo, no, l, SW, r) ;
    datao = &data[r*l+0]        ; mo =m_sub; no=r    ; load_ghost_cells(data_sub, m_sub, n_sub, l, datao, mo, no, l, EA, r) ;
    datao = &data[r*l+(n-r)]    ; mo =m_sub; no=r    ; load_ghost_cells(data_sub, m_sub, n_sub, l, datao, mo, no, l, WE, r) ;
    datao = &data[(m-r)*l+0]    ; mo =r    ; no=r    ; load_ghost_cells(data_sub, m_sub, n_sub, l, datao, mo, no, l, NE, r) ;
    datao = &data[(m-r)*l+r]    ; mo =r    ; no=n_sub; load_ghost_cells(data_sub, m_sub, n_sub, l, datao, mo, no, l, NO, r) ;
    datao = &data[(m-r)*l+(n-r)]; mo =r    ; no=r    ; load_ghost_cells(data_sub, m_sub, n_sub, l, datao, mo, no, l, NW, r) ;

  }

};

//---------------------------------------------------------------------------- 
template<typename T> 
class Convolution {
  public:
    void add(T* datao, size_t mo, size_t no, size_t lo, vector<vector<T>> &datab, size_t mb, size_t nb, size_t lb)
    {
      assert(mo==mb);
      assert(no==nb);
      assert(lo==lb);

      for (size_t i=0; i<mo; i++)
      {
        for (size_t j=0; j<no; j++)
        {
          size_t idx = i*lo+j;
          T sum = 0;
          for (size_t ibuf=0; ibuf<datab.size(); ibuf++)
            sum  += datab[ibuf][idx];

          datao[idx] = sum;
        }
      }
    }

    void B11_nw(
        T* datag, size_t mg, size_t ng, size_t lg,
        T* datai, size_t mi, size_t ni, size_t li, 
        T* datao, size_t mo, size_t no, size_t lo,
        size_t y=0, size_t x=0 //Offset 
        )
    {		
      size_t r = (ng-1)/2;
      for (size_t i=y; i<mo; i++)
      {
        for (size_t j=x; j<no; j++)
        {
          size_t idxo = i*lo+j;
          size_t offseti_y = i;
          size_t offseti_x = j;
          size_t offsetg_y = 0;
          size_t offsetg_x = 0;
          T sum = 0;
          for(size_t ii=0; ii<r-i; ii++)
          {
            for(size_t jj=0; jj<r-j; jj++)
            {
              size_t idxi = (offseti_y + ii)*li + (offseti_x + jj);
              size_t idxg = (offsetg_y + ii)*lg + (offsetg_x + jj);
              sum += datai[idxi] * datag[idxg];
            }
          }
          datao[idxo] = sum;
        }	
      }
    }

    void B11_no(
        T* datag, size_t mg, size_t ng, size_t lg,
        T* datai, size_t mi, size_t ni, size_t li, 
        T* datao, size_t mo, size_t no, size_t lo,
        size_t y=0, size_t x=0 //Offset 
        ) 
    {		
      size_t r = (ng-1)/2;
      for (size_t i=y; i<mo; i++)
      {
        for (size_t j=x; j<no; j++)
        {
          size_t idxo = i*lo+j;
          size_t offseti_y = i;
          size_t offseti_x = 0;
          size_t offsetg_y = 0;
          size_t offsetg_x = r-j;
          T sum = 0;
          for (size_t ii=0; ii<r-i; ii++)
          {
            for (size_t jj=0; jj<=r+j; jj++)
            {
              size_t idxi = (offseti_y + ii)*li + (offseti_x + jj);
              size_t idxg = (offsetg_y + ii)*lg + (offsetg_x + jj);
              sum += datai[idxi] * datag[idxg];
            }
          }
          datao[idxo] = sum;
        }	
      }
    }

    void B11_we(
        T* datag, size_t mg, size_t ng, size_t lg,
        T* datai, size_t mi, size_t ni, size_t li, 
        T* datao, size_t mo, size_t no, size_t lo,
        size_t y=0, size_t x=0 //Offset 
        ) 
    {		
      size_t r = (ng-1)/2;
      for (size_t i=y; i<mo; i++)
      {
        for (size_t j=x; j<no; j++)
        {
          size_t idxo = i*lo+j;
          size_t offseti_y = 0;
          size_t offseti_x = j;
          size_t offsetg_y = r-i;
          size_t offsetg_x = 0;
          T sum = 0;
          for (size_t ii=0; ii<=r+i; ii++)
          {
            for (size_t jj=0; jj<r-j; jj++)
            {
              size_t idxi = (offseti_y + ii)*li + (offseti_x + jj);
              size_t idxg = (offsetg_y + ii)*lg + (offsetg_x + jj);
              sum += datai[idxi] * datag[idxg];
            }
          }
          datao[idxo] = sum;
        }	
      }
    }

    void B11_aa(
        T* datag, size_t mg, size_t ng, size_t lg,
        T* datai, size_t mi, size_t ni, size_t li, 
        T* datao, size_t mo, size_t no, size_t lo,
        size_t y=0, size_t x=0 //Offset 
        ) 
    {		
      size_t r = (ng-1)/2;
      for (size_t i=y; i<mo; i++)
      {
        for (size_t j=x; j<no; j++)
        {
          size_t idxo = i*lo+j;
          size_t offseti_y = 0;
          size_t offseti_x = 0;
          size_t offsetg_y = r-i;
          size_t offsetg_x = r-j;
          T sum = 0;
          for (size_t ii=0; ii<=r+i; ii++)
          {
            for (size_t jj=0; jj<=r+j; jj++)
            {
              size_t idxi = (offseti_y + ii)*li + (offseti_x + jj);
              size_t idxg = (offsetg_y + ii)*lg + (offsetg_x + jj);
              sum += datai[idxi] * datag[idxg];
            }
          }
          datao[idxo] = sum;
        }	
      }
    }

    void B11(const Kernel2D<T> &g, Function2D<T> &fi, Function2D<T> &fo, bool use_buffers)
    {
      size_t mg = g.m;
      size_t ng = g.n;
      size_t lg = g.l;
      size_t r = (ng-1)/2;
      T* datag = g.data;

      size_t yo = 0;
      size_t xo = 0;
      size_t mo = r;
      size_t no = r;
      size_t lo = fo.l;
      T* datao = &fo.data[yo*lo+xo];

      size_t mb = mo;
      size_t nb = no;
      size_t lb = lo;

      vector<vector<T>> datab;
      if(use_buffers) datab = vector<vector<T>>(4, vector<T>(mb*lb, 0));

      if(fi.nb[NW]!=-1)
      { //NW
        size_t yi=0; size_t xi=0; size_t mi=r; size_t ni=r; Function2D<T> &ff=*fi.gr[NW]; size_t li=ff.l; T* datai=&ff.data[yi*li+xi]; // Input submatrix
        T* buf = (use_buffers)? &datab[0][0] : datao;
        B11_nw(datag, mg, ng, lg, datai, mi, ni, li, buf, mo, no, lo);
      }

      if(fi.nb[NO]!=-1)
      { //NO
        size_t yi=0; size_t xi=0; size_t mi=r; size_t ni=2*r; Function2D<T> &ff=*fi.gr[NO]; size_t li=ff.l; T* datai=&ff.data[yi*li+xi]; // Input submatrix
        T* buf = (use_buffers)? &datab[1][0] : datao;
        B11_no(datag, mg, ng, lg, datai, mi, ni, li, buf, mo, no, lo);
      }

      if(fi.nb[WE]!=-1)
      { //WE
        size_t yi=0; size_t xi=0; size_t mi=2*r; size_t ni=r; Function2D<T> &ff=*fi.gr[WE]; size_t li=ff.l; T* datai=&ff.data[yi*li+xi]; // Input submatrix
        T* buf = (use_buffers)? &datab[2][0] : datao;
        B11_we(datag, mg, ng, lg, datai, mi, ni, li, buf, mo, no, lo);
      }

      { //AA
        size_t yi=0; size_t xi=0; size_t mi=2*r; size_t ni=2*r; Function2D<T> &ff=fi; size_t li=ff.l; T* datai=&ff.data[yi*li+xi]; // Input submatrix
        T* buf = (use_buffers)? &datab[3][0] : datao;
        B11_aa(datag, mg, ng, lg, datai, mi, ni, li, buf, mo, no, lo);
      }

      if(use_buffers)
        add(datao, mo, no, lo, datab, mb, nb, lb);
    }
    //---------------------------------------------------------------------------- 

    void B12_no(
        T* datag, size_t mg, size_t ng, size_t lg,
        T* datai, size_t mi, size_t ni, size_t li, 
        T* datao, size_t mo, size_t no, size_t lo,
        size_t y=0, size_t x=0 //Offset 
        ) 
    {		
      size_t r = (ng-1)/2;
      for (size_t i=y; i<mo; i++)
      {
        for (size_t j=x; j<no; j++)
        {
          size_t idxo = i*lo+j;
          size_t offseti_y = i;
          size_t offseti_x = j;
          size_t offsetg_x = 0;
          size_t offsetg_y = 0;
          T sum = 0;
          for(size_t ii=0; ii<r-i; ii++)
          {
            for (size_t jj=0; jj<ng; jj++)
            {
              size_t idxi = (offseti_y + ii)*li + (offseti_x + jj);
              size_t idxg = (offsetg_y + ii)*lg + (offsetg_x + jj);
              sum += datai[idxi] * datag[idxg];
            }
          }
          datao[idxo] = sum;
        }
      }
    }

    void B12_aa(
        T* datag, size_t mg, size_t ng, size_t lg,
        T* datai, size_t mi, size_t ni, size_t li, 
        T* datao, size_t mo, size_t no, size_t lo,
        size_t y=0, size_t x=0 //Offset 
        ) 
    {		
      size_t r = (ng-1)/2;
      for (size_t i=y; i<mo; i++)
      {
        for (size_t j=x; j<no; j++)
        {
          size_t idxo = i*lo+j;
          size_t offseti_y = 0;
          size_t offseti_x = j;
          size_t offsetg_y = r-i;
          size_t offsetg_x = 0;
          T sum = 0;
          for (size_t ii=0; ii<=r+i; ii++)
          {
            for (size_t jj=0; jj<ng; jj++)
            {
              size_t idxi = (offseti_y + ii)*li + (offseti_x + jj);
              size_t idxg = (offsetg_y + ii)*lg + (offsetg_x + jj);
              sum += datai[idxi] * datag[idxg];
            }
          }
          datao[idxo] = sum;
        }
      }
    }

    void B12(const Kernel2D<T> &g, Function2D<T> &fi, Function2D<T> &fo, bool use_buffers)
    {
      size_t mg = g.m;
      size_t ng = g.n;
      size_t lg = g.l;
      size_t r = (ng-1)/2;
      T* datag = g.data;

      // size_t Mi = fo.m;
      size_t Ni = fo.n;

      size_t yo = 0;
      size_t xo = r;
      size_t mo = r;
      size_t no = Ni-2*r;
      size_t lo = fo.l;
      T* datao = &fo.data[yo*lo+xo];

      size_t mb = mo;
      size_t nb = no;
      size_t lb = lo;

      vector<vector<T>> datab;
      if(use_buffers) datab = vector<vector<T>>(2, vector<T>(mb*lb, 0));

      if(fi.nb[NO]!=-1)
      { //NO
        size_t yi=0; size_t xi=0; size_t mi=r; size_t ni=Ni; Function2D<T> &ff=*fi.gr[NO]; size_t li=ff.l; T* datai=&ff.data[yi*li+xi]; 
        T* buf = (use_buffers)? &datab[0][0] : datao;
        B12_no(datag, mg, ng, lg, datai, mi, ni, li, buf, mo, no, lo);
      }

      { //AA
        size_t yi=0; size_t xi=0; size_t mi=2*r; size_t ni=Ni; Function2D<T> &ff=fi; size_t li=ff.l; T* datai=&ff.data[yi*li+xi]; 
        T* buf = (use_buffers)? &datab[1][0] : datao;
        B12_aa(datag, mg, ng, lg, datai, mi, ni, li, buf, mo, no, lo);
      }

      if(use_buffers)
        add(datao, mo, no, lo, datab, mb, nb, lb);
    }

    //---------------------------------------------------------------------------- 

    void B13_no(
        T* datag, size_t mg, size_t ng, size_t lg,
        T* datai, size_t mi, size_t ni, size_t li, 
        T* datao, size_t mo, size_t no, size_t lo,
        size_t y=0, size_t x=0 //Offset 
        ) 
    {		
      size_t r = (ng-1)/2;
      for (size_t i=y; i<mo; i++)
      {
        for (size_t j=x; j<no; j++)
        {
          size_t idxo = i*lo+j;
          size_t offseti_y = i;
          size_t offseti_x = j;
          size_t offsetg_y = 0;
          size_t offsetg_x = 0;
          T sum = 0;
          for (size_t ii=0; ii<r-i; ii++)
          {
            for (size_t jj=0; jj<2*r-j; jj++)

            {
              size_t idxi = (offseti_y + ii)*li + (offseti_x + jj);
              size_t idxg = (offsetg_y + ii)*lg + (offsetg_x + jj);
              sum += datai[idxi] * datag[idxg];
            }
          }
          datao[idxo] = sum;
        }
      }
    }

    void B13_ne(
        T* datag, size_t mg, size_t ng, size_t lg,
        T* datai, size_t mi, size_t ni, size_t li, 
        T* datao, size_t mo, size_t no, size_t lo,
        size_t y=0, size_t x=0 //Offset 
        ) 
    {		
      size_t r = (ng-1)/2;
      for (size_t i=y; i<mo; i++)
      {
        for (size_t j=x; j<no; j++)
        {
          size_t idxo = i*lo+j;
          size_t offseti_y = i;
          size_t offseti_x = 0;
          size_t offsetg_y = 0;
          size_t offsetg_x = 2*r-(j);
          T sum = 0;
          for (size_t ii=0; ii<r-i; ii++)
          {
            for (size_t jj=0; jj<=(j); jj++)
            {
              size_t idxi = (offseti_y + ii)*li + (offseti_x + jj);
              size_t idxg = (offsetg_y + ii)*lg + (offsetg_x + jj);
              sum += datai[idxi] * datag[idxg];
            }
          }
          datao[idxo] = sum;
        }
      }
    }

    void B13_aa(
        T* datag, size_t mg, size_t ng, size_t lg,
        T* datai, size_t mi, size_t ni, size_t li, 
        T* datao, size_t mo, size_t no, size_t lo,
        size_t y=0, size_t x=0 //Offset 
        ) 
    {		
      size_t r = (ng-1)/2;
      for (size_t i=y; i<mo; i++)
      {
        for (size_t j=x; j<no; j++)
        {
          size_t idxo = i*lo+j;
          size_t offseti_y = 0;
          size_t offseti_x = j;
          size_t offsetg_y = r-i;
          size_t offsetg_x = 0;
          T sum = 0;
          for (size_t ii=0; ii<=r+i; ii++)
          {
            for (size_t jj=0; jj<2*r-(j); jj++)
            {
              size_t idxi = (offseti_y + ii)*li + (offseti_x + jj);
              size_t idxg = (offsetg_y + ii)*lg + (offsetg_x + jj);
              sum += datai[idxi] * datag[idxg];
            }
          }
          datao[idxo] = sum;
        }
      }
    }

    void B13_ea(
        T* datag, size_t mg, size_t ng, size_t lg,
        T* datai, size_t mi, size_t ni, size_t li, 
        T* datao, size_t mo, size_t no, size_t lo,
        size_t y=0, size_t x=0 //Offset 
        ) 
    {		
      size_t r = (ng-1)/2;
      for (size_t i=y; i<mo; i++)
      {
        for (size_t j=x; j<no; j++)
        {
          size_t idxo = i*lo+j;
          size_t offseti_y = 0;
          size_t offseti_x = 0;
          size_t offsetg_y = r-i;
          size_t offsetg_x = 2*r-j;
          T sum = 0;
          for (size_t ii=0; ii<=r+i; ii++)
          {
            for (size_t jj=0; jj<=j; jj++)
            {
              size_t idxi = (offseti_y + ii)*li + (offseti_x + jj);
              size_t idxg = (offsetg_y + ii)*lg + (offsetg_x + jj);
              sum += datai[idxi] * datag[idxg];
            }
          }
          datao[idxo] = sum;
        }
      }
    }

    void B13(const Kernel2D<T> &g, Function2D<T> &fi, Function2D<T> &fo, bool use_buffers)
    {
      size_t mg = g.m;
      size_t ng = g.n;
      size_t lg = g.l;
      size_t r = (ng-1)/2;
      T* datag = g.data;

      // size_t Mi = fo.m;
      size_t Ni = fo.n;

      size_t yo = 0;
      size_t xo = Ni-r;
      size_t mo = r;
      size_t no = r;
      size_t lo = fo.l;
      T* datao = &fo.data[yo*lo+xo];

      size_t mb = mo;
      size_t nb = no;
      size_t lb = lo;

      vector<vector<T>> datab;
      if(use_buffers) datab = vector<vector<T>>(4, vector<T>(mb*lb, 0));

      if(fi.nb[NO]!=-1)
      { //NO
        size_t yi=0; size_t xi=Ni-2*r; size_t mi=r; size_t ni=2*r; Function2D<T> &ff=*fi.gr[NO]; size_t li=ff.l; T* datai=&ff.data[yi*li+xi]; 
        T* buf = (use_buffers)? &datab[0][0] : datao;
        B13_no(datag, mg, ng, lg, datai, mi, ni, li, buf, mo, no, lo);
      }

      if(fi.nb[NE]!=-1)
      { //NE
        size_t yi=0; size_t xi=0; size_t mi=r; size_t ni=r; Function2D<T> &ff=*fi.gr[NE]; size_t li=ff.l; T* datai=&ff.data[yi*li+xi]; 
        T* buf = (use_buffers)? &datab[1][0] : datao;
        B13_ne(datag, mg, ng, lg, datai, mi, ni, li, buf, mo, no, lo);
      }

      { //AA
        size_t yi=0; size_t xi=Ni-2*r; size_t mi=2*r; size_t ni=2*r; Function2D<T> &ff=fi; size_t li=ff.l; T* datai=&ff.data[yi*li+xi]; 
        T* buf = (use_buffers)? &datab[2][0] : datao;
        B13_aa(datag, mg, ng, lg, datai, mi, ni, li, buf, mo, no, lo);
      }

      if(fi.nb[EA]!=-1)
      { //EA
        size_t yi=0; size_t xi=0; size_t mi=2*r; size_t ni=r; Function2D<T> &ff=*fi.gr[EA]; size_t li=ff.l; T* datai=&ff.data[yi*li+xi]; 
        T* buf = (use_buffers)? &datab[3][0] : datao;
        B13_ea(datag, mg, ng, lg, datai, mi, ni, li, buf, mo, no, lo);
      }

      if(use_buffers)
        add(datao, mo, no, lo, datab, mb, nb, lb);
    }
    //---------------------------------------------------------------------------- 

    void B21_we(
        T* datag, size_t mg, size_t ng, size_t lg,
        T* datai, size_t mi, size_t ni, size_t li, 
        T* datao, size_t mo, size_t no, size_t lo,
        size_t y=0, size_t x=0 //Offset 
        ) 
    {		
      size_t r = (ng-1)/2;
      for (size_t i=y; i<mo; i++)
      {
        for (size_t j=x; j<no; j++)
        {
          size_t idxo = i*lo+j;
          size_t offseti_y = i;
          size_t offseti_x = j;
          size_t offsetg_y = 0;
          size_t offsetg_x = 0;
          T sum = 0;
          for (size_t ii=0; ii<mg; ii++)
          {
            for (size_t jj=0; jj<r-j; jj++)
            {
              size_t idxi = (offseti_y + ii)*li + (offseti_x + jj);
              size_t idxg = (offsetg_y + ii)*lg + (offsetg_x + jj);
              sum += datai[idxi] * datag[idxg];
            }
          }
          datao[idxo] = sum;
        }	
      }
    }

    void B21_aa(
        T* datag, size_t mg, size_t ng, size_t lg,
        T* datai, size_t mi, size_t ni, size_t li, 
        T* datao, size_t mo, size_t no, size_t lo,
        size_t y=0, size_t x=0 //Offset 
        ) 
    {		
      size_t r = (ng-1)/2;
      for (size_t i=y; i<mo; i++)
      {
        for (size_t j=x; j<no; j++)
        {
          size_t idxo = i*lo+j;
          size_t offseti_y = i;
          size_t offseti_x = 0;
          size_t offsetg_y = 0;
          size_t offsetg_x = r-j;
          T sum = 0;
          for (size_t ii=0; ii<mg; ii++)
          {
            for (size_t jj=0; jj<=r+j; jj++)
            {
              size_t idxi = (offseti_y + ii)*li + (offseti_x + jj);
              size_t idxg = (offsetg_y + ii)*lg + (offsetg_x + jj);
              sum += datai[idxi] * datag[idxg];
            }
          }
          datao[idxo] = sum;
        }	
      }
    }

    void B21(const Kernel2D<T> &g, Function2D<T> &fi, Function2D<T> &fo, bool use_buffers)
    {
      size_t mg = g.m;
      size_t ng = g.n;
      size_t lg = g.l;
      size_t r = (ng-1)/2;
      T* datag = g.data;

      size_t Mi = fo.m;
      // size_t Ni = fo.n;

      size_t yo = r;
      size_t xo = 0;
      size_t mo = Mi-2*r;
      size_t no = r;
      size_t lo = fo.l;
      T* datao = &fo.data[yo*lo+xo];

      size_t mb = mo;
      size_t nb = no;
      size_t lb = lo;

      vector<vector<T>> datab;
      if(use_buffers) datab = vector<vector<T>>(2, vector<T>(mb*lb, 0));

      if(fi.nb[WE]!=-1)
      { //WE
        size_t yi=0; size_t xi=0; size_t mi=Mi; size_t ni=r; Function2D<T> &ff=*fi.gr[WE]; size_t li=ff.l; T* datai=&ff.data[yi*li+xi]; 
        T* buf = (use_buffers)? &datab[0][0] : datao;
        B21_we(datag, mg, ng, lg, datai, mi, ni, li, buf, mo, no, lo);
      }

      { //AA
        size_t yi=0; size_t xi=0; size_t mi=Mi; size_t ni=2*r; Function2D<T> &ff=fi; size_t li=ff.l; T* datai=&ff.data[yi*li+xi]; 
        T* buf = (use_buffers)? &datab[1][0] : datao;
        B21_aa(datag, mg, ng, lg, datai, mi, ni, li, buf, mo, no, lo);
      }

      if(use_buffers)
        add(datao, mo, no, lo, datab, mb, nb, lb);
    }

    //---------------------------------------------------------------------------- 

    void B22(
        T* datag, size_t mg, size_t ng, size_t lg,
        T* datai, size_t mi, size_t ni, size_t li, 
        T* datao, size_t mo, size_t no, size_t lo,
        size_t y=0, size_t x=0 //Offset 
        ) 
    {		
      size_t r = (ng-1)/2;
      for (size_t i=y; i<mo; i++)
      {
        for (size_t j=x; j<no; j++)
        {
          size_t idxo = i*lo+j;
          size_t offseti_y = i;
          size_t offseti_x = j;
          size_t offsetg_y = 0;
          size_t offsetg_x = 0;
          T sum = 0;
          for (size_t ii=0; ii<mg; ii++)
          {
            for (size_t jj=0; jj<ng; jj++)
            {
              size_t idxi = (offseti_y + ii)*li + (offseti_x + jj);
              size_t idxg = (offsetg_y + ii)*lg + (offsetg_x + jj);
              sum += datai[idxi] * datag[idxg];
              // printf("%f %f\n",  datai[idxi] , datag[idxg]);
            }
          }
          datao[idxo] = sum;
        }
      }
    }

    void B22(const Kernel2D<T> &g, Function2D<T> &fi, Function2D<T> &fo)
    {
      size_t mg = g.m;
      size_t ng = g.n;
      size_t lg = g.l;
      size_t r = (ng-1)/2;
      T* datag = g.data;

      size_t Mi = fo.m;
      size_t Ni = fo.n;

      size_t yo = r;
      size_t xo = r;
      size_t mo = Mi-2*r;
      size_t no = Ni-2*r;
      size_t lo = fo.l;
      T* datao = &fo.data[yo*lo+xo];

      { //AA
        size_t yi=0; size_t xi=0; size_t mi=Mi; size_t ni=Ni; Function2D<T> &ff=fi; size_t li=ff.l; T* datai=&ff.data[yi*li+xi]; 
        B22(datag, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
      }
    }

    //---------------------------------------------------------------------------- 

    void B23_aa(
        T* datag, size_t mg, size_t ng, size_t lg,
        T* datai, size_t mi, size_t ni, size_t li, 
        T* datao, size_t mo, size_t no, size_t lo,
        size_t y=0, size_t x=0 //Offset 
        ) 
    {		
      size_t r = (ng-1)/2;
      for (size_t i=y; i<mo; i++)
      {
        for (size_t j=x; j<no; j++)
        {
          size_t idxo = i*lo+j;
          size_t offseti_y = i;
          size_t offseti_x = j;
          size_t offsetg_y = 0;
          size_t offsetg_x = 0;
          T sum = 0;
          for (size_t ii=0; ii<mg; ii++)
          {
            for (size_t jj=0; jj<2*r-j; jj++)

            {
              size_t idxi = (offseti_y + ii)*li + (offseti_x + jj);
              size_t idxg = (offsetg_y + ii)*lg + (offsetg_x + jj);
              sum += datai[idxi] * datag[idxg];
            }
          }
          datao[idxo] = sum;
        }
      }
    }

    void B23_ea(
        T* datag, size_t mg, size_t ng, size_t lg,
        T* datai, size_t mi, size_t ni, size_t li, 
        T* datao, size_t mo, size_t no, size_t lo,
        size_t y=0, size_t x=0 //Offset 
        ) 
    {		
      size_t r = (ng-1)/2;
      for (size_t i=y; i<mo; i++)
      {
        for (size_t j=x; j<no; j++)
        {
          size_t idxo = i*lo+j;
          size_t offseti_y = i;
          size_t offseti_x = 0;
          size_t offsetg_y = 0;
          size_t offsetg_x = 2*r-j;
          T sum = 0;
          for (size_t ii=0; ii<mg; ii++)
          {
            for (size_t jj=0; jj<=j; jj++)
            {
              size_t idxi = (offseti_y + ii)*li + (offseti_x + jj);
              size_t idxg = (offsetg_y + ii)*lg + (offsetg_x + jj);
              sum += datai[idxi] * datag[idxg];
              // printf("%f %f\n",  datai[idxi] , datag[idxg]);
            }
          }
          datao[idxo] = sum;
        }
      }
    }

    void B23(const Kernel2D<T> &g, Function2D<T> &fi, Function2D<T> &fo, bool use_buffers)
    {
      size_t mg = g.m;
      size_t ng = g.n;
      size_t lg = g.l;
      size_t r = (ng-1)/2;
      T* datag = g.data;

      size_t Mi = fo.m;
      size_t Ni = fo.n;

      size_t yo = r;
      size_t xo = Ni-r;
      size_t mo = Mi-2*r;
      size_t no = r;
      size_t lo = fo.l;
      T* datao = &fo.data[yo*lo+xo];

      size_t mb = mo;
      size_t nb = no;
      size_t lb = lo;

      vector<vector<T>> datab;
      if(use_buffers) datab = vector<vector<T>>(2, vector<T>(mb*lb, 0));

      { //AA
        size_t yi=0; size_t xi=Ni-2*r; size_t mi=Mi; size_t ni=2*r; Function2D<T> &ff=fi; size_t li=ff.l; T* datai=&ff.data[yi*li+xi]; 
        T* buf = (use_buffers)? &datab[0][0] : datao;
        B23_aa(datag, mg, ng, lg, datai, mi, ni, li, buf, mo, no, lo);
      }

      if(fi.nb[EA]!=-1)
      { //EA
        size_t yi=0; size_t xi=0; size_t mi=Mi; size_t ni=r; Function2D<T> &ff=*fi.gr[EA]; size_t li=ff.l; T* datai=&ff.data[yi*li+xi]; 
        T* buf = (use_buffers)? &datab[1][0] : datao;
        B23_ea(datag, mg, ng, lg, datai, mi, ni, li, buf, mo, no, lo);
      }

      if(use_buffers)
        add(datao, mo, no, lo, datab, mb, nb, lb);
    }
    //---------------------------------------------------------------------------- 

    void B31_we(
        T* datag, size_t mg, size_t ng, size_t lg,
        T* datai, size_t mi, size_t ni, size_t li, 
        T* datao, size_t mo, size_t no, size_t lo, 
        size_t y=0, size_t x=0 //Offset 
        ) 
    {		
      size_t r = (ng-1)/2;
      for (size_t i=y; i<mo; i++)
      {
        for (size_t j=x; j<no; j++)
        {
          size_t idxo = i*lo+j;
          size_t offseti_y = i;
          size_t offseti_x = j;
          size_t offsetg_y = 0;
          size_t offsetg_x = 0;
          T sum = 0;
          for (size_t ii=0; ii<2*r-i; ii++)
          {
            for (size_t jj=0; jj<r-j; jj++)
            {
              size_t idxi = (offseti_y + ii)*li + (offseti_x + jj);
              size_t idxg = (offsetg_y + ii)*lg + (offsetg_x + jj);
              sum += datai[idxi] * datag[idxg];
            }
          }
          datao[idxo] = sum;
        }
      }
    }

    void B31_aa(
        T* datag, size_t mg, size_t ng, size_t lg,
        T* datai, size_t mi, size_t ni, size_t li, 
        T* datao, size_t mo, size_t no, size_t lo,
        size_t y=0, size_t x=0 //Offset 
        ) 
    {		
      size_t r = (ng-1)/2;
      for (size_t i=y; i<mo; i++)
      {
        for (size_t j=x; j<no; j++)
        {
          size_t idxo = i*lo+j;
          //
          size_t offseti_y = i;
          size_t offseti_x = 0;
          size_t offsetg_y = 0;
          size_t offsetg_x = r-j;
          T sum = 0;
          for (size_t ii=0; ii<2*r-i; ii++)
          {
            for (size_t jj=0; jj<=r+j; jj++)
            {
              size_t idxi = (offseti_y + ii)*li + (offseti_x + jj);
              size_t idxg = (offsetg_y + ii)*lg + (offsetg_x + jj);
              sum += datai[idxi] * datag[idxg];
            }
          }
          datao[idxo] = sum;
        }
      }
    }

    void B31_sw(
        T* datag, size_t mg, size_t ng, size_t lg,
        T* datai, size_t mi, size_t ni, size_t li, 
        T* datao, size_t mo, size_t no, size_t lo,
        size_t y=0, size_t x=0 //Offset 
        ) 
    {		
      size_t r = (ng-1)/2;
      for (size_t i=y; i<mo; i++)
      {
        for (size_t j=x; j<no; j++)
        {
          size_t idxo = i*lo+j;
          size_t offseti_y = 0;
          size_t offseti_x = j;
          size_t offsetg_y = 2*r-i;
          size_t offsetg_x = 0;
          T sum = 0;
          for(size_t ii=0; ii<i+1; ii++)
          {
            for(size_t jj=0; jj<r-j; jj++)
            {
              size_t idxi = (offseti_y + ii)*li + (offseti_x + jj);
              size_t idxg = (offsetg_y + ii)*lg + (offsetg_x + jj);
              sum += datai[idxi] * datag[idxg];
            }
          }
          datao[idxo] = sum;
        }
      }
    }

    void B31_so(
        T* datag, size_t mg, size_t ng, size_t lg,
        T* datai, size_t mi, size_t ni, size_t li, 
        T* datao, size_t mo, size_t no, size_t lo,
        size_t y=0, size_t x=0 //Offset 
        ) 
    {		
      size_t r = (ng-1)/2;
      for (size_t i=y; i<mo; i++)
      {
        for (size_t j=x; j<no; j++)
        {
          size_t idxo = i*lo+j;
          size_t offseti_y = 0;
          size_t offseti_x = 0;
          size_t offsetg_y = 2*r-i;
          size_t offsetg_x = r-j;
          T sum = 0;
          for(size_t ii=0; ii<i+1; ii++)
          {
            for (size_t jj=0; jj<=r+j; jj++)
            {
              size_t idxi = (offseti_y + ii)*li + (offseti_x + jj);
              size_t idxg = (offsetg_y + ii)*lg + (offsetg_x + jj);
              sum += datai[idxi] * datag[idxg];
            }
          }
          datao[idxo] = sum;
        }
      }
    }

    void B31(const Kernel2D<T> &g, Function2D<T> &fi, Function2D<T> &fo, bool use_buffers)
    {
      size_t mg = g.m;
      size_t ng = g.n;
      size_t lg = g.l;
      size_t r = (ng-1)/2;
      T* datag = g.data;

      size_t Mi = fo.m;
      // size_t Ni = fo.n;

      size_t yo = Mi-r;
      size_t xo = 0;
      size_t mo = r;
      size_t no = r;
      size_t lo = fo.l;
      T* datao = &fo.data[yo*lo+xo];

      size_t mb = mo;
      size_t nb = no;
      size_t lb = lo;
      vector<vector<T>> datab;
      if(use_buffers) datab = vector<vector<T>>(4, vector<T>(mb*lb, 0));

      if(fi.nb[WE]!=-1)
      { //WE
        size_t yi=Mi-2*r; size_t xi=0; size_t mi=2*r; size_t ni=r; Function2D<T> &ff=*fi.gr[WE]; size_t li=ff.l; T* datai=&ff.data[yi*li+xi]; 
        T* buf = (use_buffers)? &datab[0][0] : datao;
        B31_we(datag, mg, ng, lg, datai, mi, ni, li, buf, mo, no, lo);
      }

      { //AA
        size_t yi=Mi-2*r; size_t xi=0; size_t mi=2*r; size_t ni=2*r; Function2D<T> &ff=fi; size_t li=ff.l; T* datai=&ff.data[yi*li+xi]; 
        T* buf = (use_buffers)? &datab[1][0] : datao;
        B31_aa(datag, mg, ng, lg, datai, mi, ni, li, buf, mo, no, lo);
      }

      if(fi.nb[SW]!=-1)
      { //SW
        size_t yi=0; size_t xi=0; size_t mi=r; size_t ni=r; Function2D<T> &ff=*fi.gr[SW]; size_t li=ff.l; T* datai=&ff.data[yi*li+xi]; 
        T* buf = (use_buffers)? &datab[2][0] : datao;
        B31_sw(datag, mg, ng, lg, datai, mi, ni, li, buf, mo, no, lo);
      }

      if(fi.nb[SO]!=-1)
      { //SO
        size_t yi=0; size_t xi=0; size_t mi=r; size_t ni=r; Function2D<T> &ff=*fi.gr[SO]; size_t li=ff.l; T* datai=&ff.data[yi*li+xi]; 
        T* buf = (use_buffers)? &datab[3][0] : datao;
        B31_so(datag, mg, ng, lg, datai, mi, ni, li, buf, mo, no, lo);
      }

      if(use_buffers)
        add(datao, mo, no, lo, datab, mb, nb, lb);
    }

    // ---------------------------------------------------------------------------- 

    void B32_aa(
        T* datag, size_t mg, size_t ng, size_t lg,
        T* datai, size_t mi, size_t ni, size_t li, 
        T* datao, size_t mo, size_t no, size_t lo, 
        size_t y=0, size_t x=0 //Offset 
        ) 
    {		
      size_t r = (ng-1)/2;
      for (size_t i=y; i<mo; i++)
      {
        for (size_t j=x; j<no; j++)
        {
          size_t idxo = i*lo+j;
          size_t offseti_y = i;
          size_t offseti_x = j;
          size_t offsetg_y = 0;
          size_t offsetg_x = 0;
          T sum = 0;
          for (size_t ii=0; ii<2*r-i; ii++)
          {
            for (size_t jj=0; jj<ng; jj++)

            {
              size_t idxi = (offseti_y + ii)*li + (offseti_x + jj);
              size_t idxg = (offsetg_y + ii)*lg + (offsetg_x + jj);
              sum += datai[idxi] * datag[idxg];
            }
          }
          datao[idxo] = sum;
        }
      }
    }

    void B32_so(
        T* datag, size_t mg, size_t ng, size_t lg,
        T* datai, size_t mi, size_t ni, size_t li, 
        T* datao, size_t mo, size_t no, size_t lo,
        size_t y=0, size_t x=0 //Offset 
        ) 
    {		
      size_t r = (ng-1)/2;
      for (size_t i=y; i<mo; i++)
      {
        for (size_t j=x; j<no; j++)
        {
          size_t idxo = i*lo+j;
          size_t offseti_y = 0;
          size_t offseti_x = j;
          size_t offsetg_y = 2*r-i;
          size_t offsetg_x = 0;
          T sum = 0;
          for(size_t ii=0; ii<i+1; ii++)
          {
            for (size_t jj=0; jj<ng; jj++)
            {
              size_t idxi = (offseti_y + ii)*li + (offseti_x + jj);
              size_t idxg = (offsetg_y + ii)*lg + (offsetg_x + jj);
              sum += datai[idxi] * datag[idxg];
            }
          }
          datao[idxo] = sum;
        }
      }
    }

    void B32(const Kernel2D<T> &g, Function2D<T> &fi, Function2D<T> &fo, bool use_buffers)
    {
      size_t mg = g.m;
      size_t ng = g.n;
      size_t lg = g.l;
      size_t r = (ng-1)/2;
      T* datag = g.data;

      size_t Mi = fo.m;
      size_t Ni = fo.n;

      size_t yo = Mi-r;
      size_t xo = r;
      size_t mo = r;
      size_t no = Ni-2*r;
      size_t lo = fo.l;
      T* datao = &fo.data[yo*lo+xo];

      size_t mb = mo;
      size_t nb = no;
      size_t lb = lo;

      vector<vector<T>> datab;
      if(use_buffers) datab = vector<vector<T>>(2, vector<T>(mb*lb, 0));

      { //AA
        size_t yi=Mi-2*r; size_t xi=0; size_t mi=2*r; size_t ni=Ni; Function2D<T> &ff=fi; size_t li=ff.l; T* datai=&ff.data[yi*li+xi]; 
        T* buf = (use_buffers)? &datab[0][0] : datao;
        B32_aa(datag, mg, ng, lg, datai, mi, ni, li, buf, mo, no, lo);
      }

      if(fi.nb[SO]!=-1)
      { //SO
        size_t yi=0; size_t xi=0; size_t mi=r; size_t ni=Ni; Function2D<T> &ff=*fi.gr[SO]; size_t li=ff.l; T* datai=&ff.data[yi*li+xi]; 
        T* buf = (use_buffers)? &datab[1][0] : datao;
        B32_so(datag, mg, ng, lg, datai, mi, ni, li, buf, mo, no, lo);
      }

      if(use_buffers)
        add(datao, mo, no, lo, datab, mb, nb, lb);
    }

    //---------------------------------------------------------------------------- 

    void B33_aa(
        T* datag, size_t mg, size_t ng, size_t lg,
        T* datai, size_t mi, size_t ni, size_t li, 
        T* datao, size_t mo, size_t no, size_t lo, 
        size_t y=0, size_t x=0 //Offset 
        ) 
    {		
      size_t r = (ng-1)/2;
      for (size_t i=y; i<mo; i++)
      {
        for (size_t j=x; j<no; j++)
        {
          size_t idxo = i*lo+j;
          size_t offseti_y = i;
          size_t offseti_x = j;
          size_t offsetg_y = 0;
          size_t offsetg_x = 0;
          T sum = 0;
          for (size_t ii=0; ii<2*r-i; ii++)
          {
            for (size_t jj=0; jj<2*r-j; jj++)

            {
              size_t idxi = (offseti_y + ii)*li + (offseti_x + jj);
              size_t idxg = (offsetg_y + ii)*lg + (offsetg_x + jj);
              sum += datai[idxi] * datag[idxg];
            }
          }
          datao[idxo] = sum;
        }
      }
    }

    void B33_ea(
        T* datag, size_t mg, size_t ng, size_t lg,
        T* datai, size_t mi, size_t ni, size_t li, 
        T* datao, size_t mo, size_t no, size_t lo, 
        size_t y=0, size_t x=0 //Offset 
        ) 
    {		
      size_t r = (ng-1)/2;
      for (size_t i=y; i<mo; i++)
      {
        for (size_t j=x; j<no; j++)
        {
          size_t idxo = i*lo+j;
          size_t offseti_y = i;
          size_t offseti_x = 0;
          size_t offsetg_y = 0;
          size_t offsetg_x = 2*r-j;
          T sum = 0;
          for (size_t ii=0; ii<2*r-i; ii++)
          {
            for (size_t jj=0; jj<=j; jj++)
            {
              size_t idxi = (offseti_y + ii)*li + (offseti_x + jj);
              size_t idxg = (offsetg_y + ii)*lg + (offsetg_x + jj);
              sum += datai[idxi] * datag[idxg];
            }
          }
          datao[idxo] = sum;
        }
      }
    }

    void B33_so(
        T* datag, size_t mg, size_t ng, size_t lg,
        T* datai, size_t mi, size_t ni, size_t li, 
        T* datao, size_t mo, size_t no, size_t lo,
        size_t y=0, size_t x=0 //Offset 
        ) 
    {		
      size_t r = (ng-1)/2;
      for (size_t i=y; i<mo; i++)
      {
        for (size_t j=x; j<no; j++)
        {
          size_t idxo = i*lo+j;
          size_t offseti_y = 0;
          size_t offseti_x = j;
          size_t offsetg_y = 2*r-i;
          size_t offsetg_x = 0;
          T sum = 0;
          for(size_t ii=0; ii<i+1; ii++)
          {
            for (size_t jj=0; jj<2*r-j; jj++)
            {
              size_t idxi = (offseti_y + ii)*li + (offseti_x + jj);
              size_t idxg = (offsetg_y + ii)*lg + (offsetg_x + jj);
              sum += datai[idxi] * datag[idxg];
            }
          }
          datao[idxo] = sum;
        }
      }
    }

    void B33_se(
        T* datag, size_t mg, size_t ng, size_t lg,
        T* datai, size_t mi, size_t ni, size_t li, 
        T* datao, size_t mo, size_t no, size_t lo,
        size_t y=0, size_t x=0 //Offset 
        ) 
    {		
      size_t r = (ng-1)/2;
      for (size_t i=y; i<mo; i++)
      {
        for (size_t j=x; j<no; j++)
        {
          size_t idxo = i*lo+j;
          size_t offseti_y = 0;
          size_t offseti_x = 0;
          size_t offsetg_y = 2*r-i;
          size_t offsetg_x = 2*r-j;
          T sum = 0;
          for(size_t ii=0; ii<i+1; ii++)
          {
            for (size_t jj=0; jj<=j; jj++)
            {
              size_t idxi = (offseti_y + ii)*li + (offseti_x + jj);
              size_t idxg = (offsetg_y + ii)*lg + (offsetg_x + jj);
              sum += datai[idxi] * datag[idxg];
            }
          }
          datao[idxo] = sum;
        }
      }
    }

    void B33(const Kernel2D<T> &g, Function2D<T> &fi, Function2D<T> &fo, bool use_buffers)
    {
      size_t mg = g.m;
      size_t ng = g.n;
      size_t lg = g.l;
      size_t r = (ng-1)/2;
      T* datag = g.data;

      size_t Mi = fo.m;
      size_t Ni = fo.n;

      size_t yo = Mi-r;
      size_t xo = Ni-r;
      size_t mo = r;
      size_t no = r;
      size_t lo = fo.l;
      T* datao = &fo.data[yo*lo+xo];

      size_t mb = mo;
      size_t nb = no;
      size_t lb = lo;

      vector<vector<T>> datab;
      if(use_buffers) datab = vector<vector<T>>(4, vector<T>(mb*lb, 0));

      { //AA
        size_t yi=Mi-2*r; size_t xi=Ni-2*r; size_t mi=2*r; size_t ni=2*r; Function2D<T> &ff=fi; size_t li=ff.l; T* datai=&ff.data[yi*li+xi]; 
        T* buf = (use_buffers)? &datab[0][0] : datao;
        B33_aa(datag, mg, ng, lg, datai, mi, ni, li, buf, mo, no, lo);
      }

      if(fi.nb[EA]!=-1)
      { //EA
        size_t yi=Mi-2*r; size_t xi=0; size_t mi=2*r; size_t ni=r; Function2D<T> &ff=*fi.gr[EA]; size_t li=ff.l; T* datai=&ff.data[yi*li+xi]; 
        T* buf = (use_buffers)? &datab[1][0] : datao;
        B33_ea(datag, mg, ng, lg, datai, mi, ni, li, buf, mo, no, lo);
      }

      if(fi.nb[SO]!=-1)
      { //SO
        size_t yi=0; size_t xi=Ni-2*r; size_t mi=r; size_t ni=2*r; Function2D<T> &ff=*fi.gr[SO]; size_t li=ff.l; T* datai=&ff.data[yi*li+xi]; 
        T* buf = (use_buffers)? &datab[2][0] : datao;
        B33_so(datag, mg, ng, lg, datai, mi, ni, li, buf, mo, no, lo);
      }

      if(fi.nb[SE]!=-1)
      { //SE
        size_t yi=0; size_t xi=0; size_t mi=r; size_t ni=r; Function2D<T> &ff=*fi.gr[SE]; size_t li=ff.l; T* datai=&ff.data[yi*li+xi]; 
        T* buf = (use_buffers)? &datab[3][0] : datao;
        B33_se(datag, mg, ng, lg, datai, mi, ni, li, buf, mo, no, lo);
      }

      if(use_buffers)
        add(datao, mo, no, lo, datab, mb, nb, lb);
    }
};

//---------------------------------------------------------------------------- 
  template<typename T>
void compute_serial_convolution(TestFunctionConv<T> &f, const TestKernel<T> &g, TestFunction<T> &h)
{
  for (size_t i=0; i<h.m; i++)
  {
    for (size_t j=0; j<h.n; j++)
    {
      T sum = 0.0;
      for (size_t ii=0; ii<g.m; ii++)
      {
        for (size_t jj=0; jj<g.n; jj++)
        {
          size_t idxf = (i+ii)*f.l + (j+jj);
          size_t idxg = (ii  )*g.l + (jj  );
          sum += f.data[idxf]*g.data[idxg];
        }
      }
      size_t idxh = (i)*h.l+(j); 
      h.data[idxh] = sum;
    }
  }
}

//---------------------------------------------------------------------------- 
static void find_neighbours_periodic(int mpi_rank, vector<int> &nb, int mpi_grid[DIMS]) 
{
  int nx = mpi_grid[X];
  int ny = mpi_grid[Y];
  // Find neighbors
  int y = mpi_rank / nx;
  int x = mpi_rank % nx;
  int nw_y = (y - 1 + ny) % ny;
  int nw_x = (x - 1 + nx) % nx;
  int no_y = (y - 1 + ny) % ny;
  int no_x = x;
  int ne_y = (y - 1 + ny) % ny;
  int ne_x = (x + 1 + nx) % nx;
  int we_y = y;
  int we_x = (x - 1 + nx) % nx;
  int ea_y = y;
  int ea_x = (x + 1 + nx) % nx;
  int sw_y = (y + 1 + ny) % ny;
  int sw_x = (x - 1 + nx) % nx;
  int so_y = (y + 1 + ny) % ny;
  int so_x = x;
  int se_y = (y + 1 + ny) % ny;
  int se_x = (x + 1 + nx) % nx;

  nb[NW] = nw_y*nx+nw_x;
  nb[NO] = no_y*nx+no_x;
  nb[NE] = ne_y*nx+ne_x;
  nb[WE] = we_y*nx+we_x;
  nb[EA] = ea_y*nx+ea_x;
  nb[SW] = sw_y*nx+sw_x;
  nb[SO] = so_y*nx+so_x;
  nb[SE] = se_y*nx+se_x;
  nb[AA] = -1;

  // // Print list of neighbours
  // printf("[%d]: ", mpi_rank); 
  // for (int i=0; i<nb.size(); i++) printf("%2d ", nb[i]); printf("\n");
}

static void find_neighbours(int mpi_rank, vector<int> &nb, int mpi_grid[DIMS]) 
{
  int nx = mpi_grid[X];
  int ny = mpi_grid[Y];

  // Find neighbors
  int y = mpi_rank / nx;
  int x = mpi_rank % nx;
  int nw_y = (y - 1);
  int nw_x = (x - 1);
  int no_y = (y - 1);
  int no_x = x;
  int ne_y = (y - 1);
  int ne_x = (x + 1);
  int we_y = y;
  int we_x = (x - 1);
  int ea_y = y;
  int ea_x = (x + 1);
  int sw_y = (y + 1);
  int sw_x = (x - 1);
  int so_y = (y + 1);
  int so_x = x;
  int se_y = (y + 1);
  int se_x = (x + 1);

  nb[NW] = ((nw_y<0) || (nw_x<0)  )? -1: nw_y*nx+nw_x;
  nb[NO] = ((no_y<0)              )? -1: no_y*nx+no_x;
  nb[NE] = ((ne_y<0) || (ne_x>=nx))? -1: ne_y*nx+ne_x;

  nb[WE] = ((we_x<0)  )? -1: we_y*nx+we_x;
  nb[EA] = ((ea_x>=nx))? -1: ea_y*nx+ea_x;

  nb[SW] = ((sw_y>=ny) || (sw_x<0)  )? -1: sw_y*nx+sw_x;
  nb[SO] = ((so_y>=ny)              )? -1: so_y*nx+so_x;
  nb[SE] = ((se_y>=ny) || (se_x>=nx))? -1: se_y*nx+se_x;
  nb[AA] = -1;

  // Print list of neighbours
  // printf("[%d]: ", mpi_rank); 
  // for (int i=0; i<nb.size(); i++) printf("%+2d ", nb[i]);
  // printf("\n");
}



  template<typename T>
void rank_matrix_to_submatrices(
    const TestKernel<T> &g, 
    TestFunction<T> &f,
    vector< TestFunction<T>* > &f_sub,
    vector< TestFunction<T>* > &h_sub,
    size_t &f_sub_mem, size_t &h_sub_mem, 
    size_t ms, size_t ns, size_t ny, size_t nx)
{
  for (size_t i=0; i<ny; i++)
  {
    for (size_t j=0; j<nx; j++)
    {
      size_t idx = i*nx+j;
      f_sub[idx] = new TestFunction<T>(ms+2*g.r, ns+2*g.r, InitType::NONE);
      h_sub[idx] = new TestFunction<T>(ms, ns, InitType::ZERO);
      f_sub_mem += f_sub[idx]->get_mem_size();
      h_sub_mem += h_sub[idx]->get_mem_size();

      size_t offsetf_i = i*ms;
      size_t offsetf_j = j*ns;

      for (size_t ii=0; ii<f_sub[idx]->m; ii++)
      {
        for (size_t jj=0; jj<f_sub[idx]->n; jj++)
        {
          size_t idxs = ii*f_sub[idx]->l + jj;
          size_t idxf = (offsetf_i + ii)*f.l + (offsetf_j+jj);
          f_sub[idx]->data[idxs]  = f.data[idxf];
        }
      }
    }
  }
}

  template<typename T>
void submatrices_into_rank_matrix(
    TestFunction<T> &h, 
    vector<TestFunction<T>* > &h_sub, 
    size_t ms, size_t ns, size_t ny, size_t nx)
{
  for (size_t i=0; i<ny; i++)
  {
    for (size_t j=0; j<nx; j++)
    {
      size_t idx = i*nx+j;
      size_t offseth_i = i*ms;
      size_t offseth_j = j*ns;

      for (size_t ii=0; ii<h_sub[idx]->m; ii++)
      {
        for (size_t jj=0; jj<h_sub[idx]->n; jj++)
        {
          size_t idxh = (offseth_i+ii)*h.l + (offseth_j+jj); // rank matrix
          size_t idxs = (ii)*h_sub[idx]->l + (jj);           // sub  matrix
          h.data[idxh] = h_sub[idx]->data[idxs];
        }
      }
    }
  }
}

  template<typename T>
void compute_convolution_sub(
    T* datag, size_t mg, size_t ng, size_t lg,
    T* datai, size_t mi, size_t ni, size_t li, 
    T* datao, size_t mo, size_t no, size_t lo)
{
  size_t r = mg/2;
  for (size_t i=0; i<mo; i++)
  {
    for (size_t j=0; j<no; j++)
    {
      size_t idxh = i*lo+j; 
      size_t offset_i = i;
      size_t offset_j = j;
      T sum = 0;
      for (size_t ii=0; ii<mg; ii++)
      {
        for (size_t jj=0; jj<ng; jj++)
        {
          size_t idxf = (offset_i+ii)*li + (offset_j+jj);
          size_t idxg = ii*lg+ jj;
          sum += datai[idxf] * datag[idxg];
        }
      }
      datao[idxh] = sum;
    }
  }
}

#endif
