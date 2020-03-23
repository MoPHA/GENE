#include "convolution_core.h"
#include <starpu.h>
// ============================================================================ 
// Data handlers
// ============================================================================ 

template<typename T>
class DH_ghost_cells
{
  public:
    vector<starpu_data_handle_t> s; 
    vector<starpu_data_handle_t> r; 

    void register_views(TestFunction<T> &f)
    {
      s.resize(n_neighbors);
      r.resize(n_neighbors);
      {
        for (int i=0; i<n_neighbors; i++)
        {
          if(f.nb[i] == -1) continue;
          starpu_matrix_data_register(&s[i], STARPU_MAIN_RAM, (uintptr_t)f.gs[i]->data, f.gs[i]->l, f.gs[i]->n, f.gs[i]->m, sizeof(f.gs[i]->data[0]));
          starpu_matrix_data_register(&r[i], STARPU_MAIN_RAM, (uintptr_t)f.gr[i]->data, f.gr[i]->l, f.gr[i]->n, f.gr[i]->m, sizeof(f.gr[i]->data[0]));
        }
      }
    }

    void unregister_views(TestFunction<T> &f)
    {
      for (int i=0; i<n_neighbors; i++)
      {
        if(f.nb[i] == -1) continue;
        starpu_data_unregister(r[i]);
        starpu_data_unregister(s[i]);
      }
    }
};

template<typename T>
class DH_kernel
{
  public:
    starpu_data_handle_t full; 

    void register_views(const Kernel2D<T> &g)
    {
      starpu_matrix_data_register(&full, STARPU_MAIN_RAM, (uintptr_t)g.data, g.l, g.n, g.m, sizeof(g.data[0]));
    }

    void unregister_views()
    {
      starpu_data_unregister(full);
    }
};

template<typename T>
class DH_input_mem_conv
{
  public:
    starpu_data_handle_t full; 

    // Input blocks required to compute convolution in each output block
    vector<starpu_data_handle_t> input_block_conv;

    // Input blocks required to compute convolution in each output AA_block
    vector<starpu_data_handle_t> input_AA_block_conv;

    // ---------------------------------------------------------------------------- 
    // Register views
    // ---------------------------------------------------------------------------- 
    void register_views(TestFunction<T> &f, const Kernel2D<T> &g, size_t msub, size_t nsub, int ny, int nx)
    {
      int r = g.r;

      starpu_matrix_data_register(&full, STARPU_MAIN_RAM, (uintptr_t)f.data, f.l, f.n, f.m, sizeof(f.data[0]));

      input_block_conv.resize(n_blocks);
      {
        size_t m = f.m;
        size_t n = f.n;
        for (int iblock=0; iblock<n_blocks; iblock++)
        {
          size_t yi; size_t xi; size_t mi; size_t ni; size_t li; T* datai;
          if (iblock == NW) { yi=0     ; xi=0     ; mi=2*r ; ni=2*r ; li=f.l ; }
          else if (iblock == NO) { yi=0     ; xi=0     ; mi=2*r ; ni=n   ; li=f.l ; }
          else if (iblock == NE) { yi=0     ; xi=n-2*r ; mi=2*r ; ni=2*r ; li=f.l ; }
          else if (iblock == WE) { yi=0     ; xi=0     ; mi=m   ; ni=2*r ; li=f.l ; }
          else if (iblock == AA) { yi=0     ; xi=0     ; mi=m   ; ni=n   ; li=f.l ; }
          else if (iblock == EA) { yi=0     ; xi=n-2*r ; mi=m   ; ni=2*r ; li=f.l ; }
          else if (iblock == SW) { yi=m-2*r ; xi=0     ; mi=2*r ; ni=2*r ; li=f.l ; }
          else if (iblock == SO) { yi=m-2*r ; xi=0     ; mi=2*r ; ni=n   ; li=f.l ; }
          else if (iblock == SE) { yi=m-2*r ; xi=n-2*r ; mi=2*r ; ni=2*r ; li=f.l ; }

          datai=&f.data[yi*li+xi]; 

          starpu_matrix_data_register(&input_block_conv[iblock], STARPU_MAIN_RAM, (uintptr_t)datai, li, ni, mi, sizeof(datai[0]));
        }
      }

      input_AA_block_conv.resize(ny*nx);
      {
        for (size_t i=0; i<ny; i++)
        {
          for (size_t j=0; j<nx; j++)
          {
            size_t idx = i*nx+j;
            size_t yo=r+(i*msub);
            size_t xo=r+(nsub*j);
            size_t mo=msub; 
            size_t no=nsub; 
            size_t lo=f.l;
            size_t yi=yo-r;
            size_t xi=xo-r;
            size_t mi=mo+2*r;
            size_t ni=no+2*r;
            size_t li=f.l;
            T* datai=&f.data[yi*li+xi]; 
            starpu_matrix_data_register(&input_AA_block_conv[idx], STARPU_MAIN_RAM, (uintptr_t)datai, li, ni, mi, sizeof(datai[0]));
          }
        }
      }
    }

    void unregister_views()
    {
      starpu_data_unregister(full);
      for (int i=0; i<input_block_conv.size(); i++) starpu_data_unregister(input_block_conv[i]);
      for (int i=0; i<input_AA_block_conv.size(); i++) starpu_data_unregister(input_AA_block_conv[i]);
    }
};


template<typename T>
class DH_output_mem_conv
{
  public:
    starpu_data_handle_t full;

    // The memory is divided in 9 regions: NW, NO, NE, WE, AA, EA, SW, SO, SE
    vector<starpu_data_handle_t> block;

    // Blocks in which AA block is divided
    vector<starpu_data_handle_t> AA_block;

    // Buffers: To allow parallelization of tasks
    vector<starpu_data_handle_t> buf_nw;
    vector<starpu_data_handle_t> buf_no;
    vector<starpu_data_handle_t> buf_ne;
    vector<starpu_data_handle_t> buf_we;
    vector<starpu_data_handle_t> buf_ea;
    vector<starpu_data_handle_t> buf_sw;
    vector<starpu_data_handle_t> buf_so;
    vector<starpu_data_handle_t> buf_se;

    // Create buffers for addition
    vector<TestFunction<T>*> h_nw;
    vector<TestFunction<T>*> h_no;
    vector<TestFunction<T>*> h_ne;
    vector<TestFunction<T>*> h_we;
    vector<TestFunction<T>*> h_ea;
    vector<TestFunction<T>*> h_sw;
    vector<TestFunction<T>*> h_so;
    vector<TestFunction<T>*> h_se;

    // ---------------------------------------------------------------------------- 
    // Register views
    // ---------------------------------------------------------------------------- 
    void register_views(TestFunction<T> &h, const Kernel2D<T> &g, size_t msub, size_t nsub, size_t ny, size_t nx)
    {
      int r = g.r;
      size_t m = h.m;
      size_t n = h.n;

      starpu_matrix_data_register(&full, STARPU_MAIN_RAM, (uintptr_t)h.data, h.l, h.n, h.m, sizeof(h.data[0]));

      block.resize(n_blocks);
      {
        for (int iblock=0; iblock<n_neighbors; iblock++)
        {
          size_t yo; size_t xo; size_t mo; size_t no; size_t lo; T* datao;
          if (iblock == NW) { yo = 0   ; xo = 0   ; mo = r     ; no = r     ; lo = h.l ; }
          else if (iblock == NO) { yo = 0   ; xo = r   ; mo = r     ; no = n-2*r ; lo = h.l ; }
          else if (iblock == NE) { yo = 0   ; xo = n-r ; mo = r     ; no = r     ; lo = h.l ; }
          else if (iblock == WE) { yo = r   ; xo = 0   ; mo = m-2*r ; no = r     ; lo = h.l ; }
          else if (iblock == AA) { yo = r   ; xo = r   ; mo = m-2*r ; no = n-2*r ; lo = h.l ; }
          else if (iblock == EA) { yo = r   ; xo = n-r ; mo = m-2*r ; no = r     ; lo = h.l ; }
          else if (iblock == SW) { yo = m-r ; xo = 0   ; mo = r     ; no = r     ; lo = h.l ; }
          else if (iblock == SO) { yo = m-r ; xo = r   ; mo = r     ; no = n-2*r ; lo = h.l ; }
          else if (iblock == SE) { yo = m-r ; xo = n-r ; mo = r     ; no = r     ; lo = h.l ; }
          datao = &h.data[yo*lo+xo];
          starpu_matrix_data_register(&block[iblock], STARPU_MAIN_RAM, (uintptr_t)datao, lo, no, mo, sizeof(datao[0]));
        }
      }

      AA_block.resize(ny*nx);
      {
        for (size_t i=0; i<ny; i++)
        {
          for (size_t j=0; j<nx; j++)
          {
            size_t idx = i*nx+j;
            size_t yo=r+(i*msub);
            size_t xo=r+(nsub*j);
            size_t mo=msub; 
            size_t no=nsub; 
            size_t lo=h.l;
            T* datao=&h.data[yo*lo+xo]; 
            starpu_matrix_data_register(&AA_block[idx], STARPU_MAIN_RAM, (uintptr_t)datao, lo, no, mo, sizeof(datao[0]));
          }
        }
      }

      {
        buf_nw.resize(4);
        buf_no.resize(2);
        buf_ne.resize(4);
        buf_we.resize(2);
        buf_ea.resize(2);
        buf_sw.resize(4);
        buf_so.resize(2);
        buf_se.resize(4);

        // Create buffers for addition
        h_nw.resize(4, NULL);
        h_no.resize(2, NULL);
        h_ne.resize(4, NULL);
        h_we.resize(2, NULL);
        h_ea.resize(2, NULL);
        h_sw.resize(4, NULL);
        h_so.resize(2, NULL);
        h_se.resize(4, NULL);

        for (int i=0; i<h_nw.size(); i++) h_nw[i] = new TestFunction<T>(r, r, InitType::ZERO);
        for (int i=0; i<h_no.size(); i++) h_no[i] = new TestFunction<T>(r, n-2*r, InitType::ZERO);
        for (int i=0; i<h_ne.size(); i++) h_ne[i] = new TestFunction<T>(r, r, InitType::ZERO);
        for (int i=0; i<h_we.size(); i++) h_we[i] = new TestFunction<T>(m-2*r, r, InitType::ZERO);
        for (int i=0; i<h_ea.size(); i++) h_ea[i] = new TestFunction<T>(m-2*r, r, InitType::ZERO);
        for (int i=0; i<h_sw.size(); i++) h_sw[i] = new TestFunction<T>(r, r, InitType::ZERO);
        for (int i=0; i<h_so.size(); i++) h_so[i] = new TestFunction<T>(r, n-2*r, InitType::ZERO);
        for (int i=0; i<h_se.size(); i++) h_se[i] = new TestFunction<T>(r, r,  InitType::ZERO);

        // Register buffers
        for (int i=0; i<h_nw.size(); i++) starpu_matrix_data_register(&buf_nw[i], STARPU_MAIN_RAM, (uintptr_t)h_nw[i]->data, h_nw[i]->l, h_nw[i]->n, h_nw[i]->m, sizeof(h_nw[i]->data[0]));
        for (int i=0; i<h_no.size(); i++) starpu_matrix_data_register(&buf_no[i], STARPU_MAIN_RAM, (uintptr_t)h_no[i]->data, h_no[i]->l, h_no[i]->n, h_no[i]->m, sizeof(h_no[i]->data[0]));
        for (int i=0; i<h_ne.size(); i++) starpu_matrix_data_register(&buf_ne[i], STARPU_MAIN_RAM, (uintptr_t)h_ne[i]->data, h_ne[i]->l, h_ne[i]->n, h_ne[i]->m, sizeof(h_ne[i]->data[0]));
        for (int i=0; i<h_we.size(); i++) starpu_matrix_data_register(&buf_we[i], STARPU_MAIN_RAM, (uintptr_t)h_we[i]->data, h_we[i]->l, h_we[i]->n, h_we[i]->m, sizeof(h_we[i]->data[0]));
        for (int i=0; i<h_ea.size(); i++) starpu_matrix_data_register(&buf_ea[i], STARPU_MAIN_RAM, (uintptr_t)h_ea[i]->data, h_ea[i]->l, h_ea[i]->n, h_ea[i]->m, sizeof(h_ea[i]->data[0]));
        for (int i=0; i<h_sw.size(); i++) starpu_matrix_data_register(&buf_sw[i], STARPU_MAIN_RAM, (uintptr_t)h_sw[i]->data, h_sw[i]->l, h_sw[i]->n, h_sw[i]->m, sizeof(h_sw[i]->data[0]));
        for (int i=0; i<h_so.size(); i++) starpu_matrix_data_register(&buf_so[i], STARPU_MAIN_RAM, (uintptr_t)h_so[i]->data, h_so[i]->l, h_so[i]->n, h_so[i]->m, sizeof(h_so[i]->data[0]));
        for (int i=0; i<h_se.size(); i++) starpu_matrix_data_register(&buf_se[i], STARPU_MAIN_RAM, (uintptr_t)h_se[i]->data, h_se[i]->l, h_se[i]->n, h_se[i]->m, sizeof(h_se[i]->data[0]));
      }
    }

    void unregister_views()
    {
      starpu_data_unregister(full);

      for (int i=0; i<block.size(); i++) starpu_data_unregister(block[i]);

      for (int i=0; i<AA_block.size(); i++) starpu_data_unregister(AA_block[i]);

      for (int i=0; i<h_nw.size(); i++) { starpu_data_unregister(buf_nw[i]); delete h_nw[i]; }
      for (int i=0; i<h_no.size(); i++) { starpu_data_unregister(buf_no[i]); delete h_no[i]; }
      for (int i=0; i<h_ne.size(); i++) { starpu_data_unregister(buf_ne[i]); delete h_ne[i]; }
      for (int i=0; i<h_we.size(); i++) { starpu_data_unregister(buf_we[i]); delete h_we[i]; }
      for (int i=0; i<h_ea.size(); i++) { starpu_data_unregister(buf_ea[i]); delete h_ea[i]; }
      for (int i=0; i<h_sw.size(); i++) { starpu_data_unregister(buf_sw[i]); delete h_sw[i]; }
      for (int i=0; i<h_so.size(); i++) { starpu_data_unregister(buf_so[i]); delete h_so[i]; }
      for (int i=0; i<h_se.size(); i++) { starpu_data_unregister(buf_se[i]); delete h_se[i]; }
    }
};


template<typename T>
class DH_Global
{
  public:
    vector<starpu_data_handle_t> block;  //Local views 

    void register_views(TestFunction<T> &f, int mpi_grid[DIMS], int my_rank, int m, int n)
    {
      if(my_rank == 0)
      {

        block.resize(mpi_grid[Y]*mpi_grid[X]);

        for (int i=0; i<mpi_grid[Y]; i++)
        {
          for (int j=0; j<mpi_grid[X]; j++)
          {
            int mpi_rank = i*mpi_grid[X]+j; 

            size_t offset_i = i * m;
            size_t offset_j = j * n;
            size_t offset  = (offset_i)*f.l + (offset_j);
            T* datao = &f.data[offset];
            starpu_matrix_data_register(&block[mpi_rank], STARPU_MAIN_RAM, (uintptr_t)datao, f.l, n, m, sizeof(datao[0]));
          }
        }
      }
    }

    void unregister_views()
    {
      for(int i=0; i<block.size(); i++) starpu_data_unregister(block[i]);
    }
};



//============================================================================ 
// Codeletes and function definions
//============================================================================ 

static Convolution<double> convolve;

void get_data(
    double *&fg, size_t &mg, size_t &ng, size_t &lg,
    double *&fi, size_t &mi, size_t &ni, size_t &li, 
    double *&fo, size_t &mo, size_t &no, size_t &lo,
    void *buffers[], void *cl_arg)
{
  fo = (double*)STARPU_MATRIX_GET_PTR(buffers[0]);
  no = (unsigned)STARPU_MATRIX_GET_NX(buffers[0]);
  mo = (unsigned)STARPU_MATRIX_GET_NY(buffers[0]);
  lo = (unsigned)STARPU_MATRIX_GET_LD(buffers[0]);

  fi = (double*)STARPU_MATRIX_GET_PTR(buffers[1]);
  ni = (unsigned)STARPU_MATRIX_GET_NX(buffers[1]);
  mi = (unsigned)STARPU_MATRIX_GET_NY(buffers[1]);
  li = (unsigned)STARPU_MATRIX_GET_LD(buffers[1]);

  fg = (double*)STARPU_MATRIX_GET_PTR(buffers[2]);
  ng = (unsigned)STARPU_MATRIX_GET_NX(buffers[2]);
  mg = (unsigned)STARPU_MATRIX_GET_NY(buffers[2]);
  lg = (unsigned)STARPU_MATRIX_GET_LD(buffers[2]);
}

void add_2buffers(void *buffers[], void *cl_arg)
{
  double *fo=NULL; size_t  no, mo, lo; 
  double *fi1=NULL; size_t  ni1, mi1, li1; 
  double *fi2=NULL; size_t  ni2, mi2, li2; 

  fo = (double*)STARPU_MATRIX_GET_PTR(buffers[0]);
  no = (unsigned)STARPU_MATRIX_GET_NX(buffers[0]);
  mo = (unsigned)STARPU_MATRIX_GET_NY(buffers[0]);
  lo = (unsigned)STARPU_MATRIX_GET_LD(buffers[0]);

  fi1 = (double*)STARPU_MATRIX_GET_PTR(buffers[1]);
  ni1 = (unsigned)STARPU_MATRIX_GET_NX(buffers[1]);
  mi1 = (unsigned)STARPU_MATRIX_GET_NY(buffers[1]);
  li1 = (unsigned)STARPU_MATRIX_GET_LD(buffers[1]);

  fi2 = (double*)STARPU_MATRIX_GET_PTR(buffers[2]);
  ni2 = (unsigned)STARPU_MATRIX_GET_NX(buffers[2]);
  mi2 = (unsigned)STARPU_MATRIX_GET_NY(buffers[2]);
  li2 = (unsigned)STARPU_MATRIX_GET_LD(buffers[2]);

  for (size_t i=0; i<mo; i++) {
    for (size_t j=0; j<no; j++) {
      fo[i*lo+j] = fi1[i*li1+j]+ fi2[i*li2+j];
    }
  }
}

void add_4buffers(void *buffers[], void *cl_arg)
{
  double *fo=NULL; size_t  no, mo, lo; 
  double *fi1=NULL; size_t  ni1, mi1, li1; 
  double *fi2=NULL; size_t  ni2, mi2, li2; 
  double *fi3=NULL; size_t  ni3, mi3, li3; 
  double *fi4=NULL; size_t  ni4, mi4, li4; 

  fo = (double*)STARPU_MATRIX_GET_PTR(buffers[0]);
  no = (unsigned)STARPU_MATRIX_GET_NX(buffers[0]);
  mo = (unsigned)STARPU_MATRIX_GET_NY(buffers[0]);
  lo = (unsigned)STARPU_MATRIX_GET_LD(buffers[0]);

  fi1 = (double*)STARPU_MATRIX_GET_PTR(buffers[1]);
  ni1 = (unsigned)STARPU_MATRIX_GET_NX(buffers[1]);
  mi1 = (unsigned)STARPU_MATRIX_GET_NY(buffers[1]);
  li1 = (unsigned)STARPU_MATRIX_GET_LD(buffers[1]);

  fi2 = (double*)STARPU_MATRIX_GET_PTR(buffers[2]);
  ni2 = (unsigned)STARPU_MATRIX_GET_NX(buffers[2]);
  mi2 = (unsigned)STARPU_MATRIX_GET_NY(buffers[2]);
  li2 = (unsigned)STARPU_MATRIX_GET_LD(buffers[2]);

  fi3 = (double*)STARPU_MATRIX_GET_PTR(buffers[3]);
  ni3 = (unsigned)STARPU_MATRIX_GET_NX(buffers[3]);
  mi3 = (unsigned)STARPU_MATRIX_GET_NY(buffers[3]);
  li3 = (unsigned)STARPU_MATRIX_GET_LD(buffers[3]);

  fi4 = (double*)STARPU_MATRIX_GET_PTR(buffers[4]);
  ni4 = (unsigned)STARPU_MATRIX_GET_NX(buffers[4]);
  mi4 = (unsigned)STARPU_MATRIX_GET_NY(buffers[4]);
  li4 = (unsigned)STARPU_MATRIX_GET_LD(buffers[4]);

  for (size_t i=0; i<mo; i++) {
    for (size_t j=0; j<no; j++) {
      fo[i*lo+j] = fi1[i*li1+j]+ fi2[i*li2+j] + fi3[i*li3+j] + fi4[i*li4+j];
    }
  }
}

// ---------------------------------------------------------------------------- 
void convolve_B11_nw(void *buffers[], void *cl_arg)
{
  double *datak=NULL; double *datai=NULL; double *datao=NULL;
  size_t ng, mg, lg, ni, mi, li, no, mo, lo; 
  get_data(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo, buffers, cl_arg);

  size_t r = (ng-1)/2;
  size_t yi=0; size_t xi=0;  mi=r;  ni=r; 
  datai+=(yi*li+xi); 
  convolve.B11_nw(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
}

void convolve_B11_no(void *buffers[], void *cl_arg)
{
  double *datak, *datai, *datao;  size_t ng, mg, lg, ni, mi, li, no, mo, lo; 
  get_data(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo, buffers, cl_arg);

  size_t r = (ng-1)/2;
  size_t yi=0; size_t xi=0;  mi=r;  ni=2*r; 
  datai+=(yi*li+xi); 
  convolve.B11_no(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
}

void convolve_B11_we(void *buffers[], void *cl_arg)
{
  double *datak, *datai, *datao;  size_t ng, mg, lg, ni, mi, li, no, mo, lo; 
  get_data(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo, buffers, cl_arg);

  size_t r = (ng-1)/2; 
  size_t yi=0; size_t xi=0;  mi=2*r;  ni=r;
  datai+=(yi*li+xi); 
  convolve.B11_we(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
}

void convolve_B11_aa(void *buffers[], void *cl_arg)
{
  double *datak, *datai, *datao;  size_t ng, mg, lg, ni, mi, li, no, mo, lo; 
  get_data(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo, buffers, cl_arg);

  size_t r = (ng-1)/2;
  size_t yi=0; size_t xi=0;  mi=2*r;  ni=2*r; 
  datai+=(yi*li+xi); 
  convolve.B11_aa(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
}

// ---------------------------------------------------------------------------- 
void convolve_B21_we(void *buffers[], void *cl_arg)
{
  double *datak, *datai, *datao;  size_t ng, mg, lg, ni, mi, li, no, mo, lo; 
  get_data(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo, buffers, cl_arg);

  size_t r = (ng-1)/2;
  size_t yi=0; size_t xi=0;  mi=mi;  ni=r; 
  datai+=(yi*li+xi); 
  convolve.B21_we(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
}

void convolve_B21_aa(void *buffers[], void *cl_arg)
{
  double *datak, *datai, *datao;  size_t ng, mg, lg, ni, mi, li, no, mo, lo; 
  get_data(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo, buffers, cl_arg);

  size_t r = (ng-1)/2;
  size_t yi=0; size_t xi=0;  mi=mi;  ni=2*r; 
  datai+=(yi*li+xi); 
  convolve.B21_aa(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
}
// ---------------------------------------------------------------------------- 

void convolve_B31_we(void *buffers[], void *cl_arg)
{
  double *datak, *datai, *datao;  size_t ng, mg, lg, ni, mi, li, no, mo, lo; 
  get_data(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo, buffers, cl_arg);

  size_t r = (ng-1)/2;
  size_t yi=mi-2*r; size_t xi=0; mi=2*r; ni=r;
  datai+=(yi*li+xi); 
  convolve.B31_we(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
}

void convolve_B31_aa(void *buffers[], void *cl_arg)
{
  double *datak, *datai, *datao;  size_t ng, mg, lg, ni, mi, li, no, mo, lo; 
  get_data(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo, buffers, cl_arg);

  size_t r = (ng-1)/2;
  size_t yi=mi-2*r; size_t xi=0;  mi=2*r;  ni=2*r;
  datai+=(yi*li+xi); 
  convolve.B31_aa(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
}

void convolve_B31_sw(void *buffers[], void *cl_arg)
{
  double *datak, *datai, *datao;  size_t ng, mg, lg, ni, mi, li, no, mo, lo; 
  get_data(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo, buffers, cl_arg);

  size_t r = (ng-1)/2;
  size_t yi=0; size_t xi=0;  mi=r;  ni=r;
  datai+=(yi*li+xi); 
  convolve.B31_sw(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
}

void convolve_B31_so(void *buffers[], void *cl_arg)
{
  double *datak, *datai, *datao;  size_t ng, mg, lg, ni, mi, li, no, mo, lo; 
  get_data(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo, buffers, cl_arg);

  size_t r = (ng-1)/2;
  size_t yi=0; size_t xi=0; mi=r; ni=r;
  datai+=(yi*li+xi); 
  convolve.B31_so(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
}


// ---------------------------------------------------------------------------- 
void convolve_B12_no(void *buffers[], void *cl_arg)
{
  double *datak, *datai, *datao;  size_t ng, mg, lg, ni, mi, li, no, mo, lo; 
  get_data(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo, buffers, cl_arg);

  size_t r = (ng-1)/2;
  size_t yi=0; size_t xi=0; mi=r; ni=ni; 
  datai+=(yi*li+xi); 
  convolve.B12_no(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
}

void convolve_B12_aa(void *buffers[], void *cl_arg)
{
  double *datak, *datai, *datao;  size_t ng, mg, lg, ni, mi, li, no, mo, lo; 
  get_data(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo, buffers, cl_arg);

  size_t r = (ng-1)/2;
  size_t yi=0; size_t xi=0; mi=2*r;  ni=ni;
  datai+=(yi*li+xi); 
  convolve.B12_aa(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
}

// ---------------------------------------------------------------------------- 

#if B22_ONE_TASK
void convolve_B22_aa(void *buffers[], void *cl_arg)
{
  double *datak, *datai, *datao;  size_t ng, mg, lg, ni, mi, li, no, mo, lo; 
  get_data(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo, buffers, cl_arg);
  convolve.B22(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
}

#else

void compute_convolution_cpu_func(void *buffers[], void *_args)
{
  double *fo, *fi, *fg;
  size_t no, mo, lo, ni, mi, li, ng, mg, lg;

  fo = (double*)STARPU_MATRIX_GET_PTR(buffers[0]);
  no = (unsigned)STARPU_MATRIX_GET_NX(buffers[0]);
  mo = (unsigned)STARPU_MATRIX_GET_NY(buffers[0]);
  lo = (unsigned)STARPU_MATRIX_GET_LD(buffers[0]);

  fi = (double*)STARPU_MATRIX_GET_PTR(buffers[1]);
  ni = (unsigned)STARPU_MATRIX_GET_NX(buffers[1]);
  mi = (unsigned)STARPU_MATRIX_GET_NY(buffers[1]);
  li = (unsigned)STARPU_MATRIX_GET_LD(buffers[1]);

  fg = (double*)STARPU_MATRIX_GET_PTR(buffers[2]);
  ng = (unsigned)STARPU_MATRIX_GET_NX(buffers[2]);
  mg = (unsigned)STARPU_MATRIX_GET_NY(buffers[2]);
  lg = (unsigned)STARPU_MATRIX_GET_LD(buffers[2]);

  compute_convolution_sub( fg, mg, ng, lg, fi, mi, ni, li, fo, mo, no, lo);
}

extern void compute_convolution_gpu_func(void *buffers[], void *cl_arg);

#endif

// ---------------------------------------------------------------------------- 

void convolve_B32_aa(void *buffers[], void *cl_arg)
{
  double *datak, *datai, *datao;  size_t ng, mg, lg, ni, mi, li, no, mo, lo; 
  get_data(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo, buffers, cl_arg);

  size_t r = (ng-1)/2;
  size_t yi=mi-2*r; size_t xi=0;  mi=2*r;  ni=ni;
  datai+=(yi*li+xi); 
  convolve.B32_aa(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
}

void convolve_B32_so(void *buffers[], void *cl_arg)
{
  double *datak, *datai, *datao;  size_t ng, mg, lg, ni, mi, li, no, mo, lo; 
  get_data(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo, buffers, cl_arg);

  size_t r = (ng-1)/2;
  size_t yi=0; size_t xi=0;  mi=r;  ni=ni;
  datai+=(yi*li+xi); 
  convolve.B32_so(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
}

// ---------------------------------------------------------------------------- 

void convolve_B13_no(void *buffers[], void *cl_arg)
{
  double *datak, *datai, *datao;  size_t ng, mg, lg, ni, mi, li, no, mo, lo; 
  get_data(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo, buffers, cl_arg);

  size_t r = (ng-1)/2;
  size_t yi=0; size_t xi=ni-2*r;  mi=r;  ni=2*r; 
  datai+=(yi*li+xi); 
  convolve.B13_no(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
}

void convolve_B13_ne(void *buffers[], void *cl_arg)
{
  double *datak, *datai, *datao;  size_t ng, mg, lg, ni, mi, li, no, mo, lo; 
  get_data(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo, buffers, cl_arg);

  size_t r = (ng-1)/2;
  size_t yi=0; size_t xi=0;  mi=r; ni=r;
  datai+=(yi*li+xi); 
  convolve.B13_ne(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
}

void convolve_B13_aa(void *buffers[], void *cl_arg)
{
  double *datak, *datai, *datao;  size_t ng, mg, lg, ni, mi, li, no, mo, lo; 
  get_data(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo, buffers, cl_arg);

  size_t r = (ng-1)/2;
  size_t yi=0; size_t xi=ni-2*r;  mi=2*r;  ni=2*r;
  datai+=(yi*li+xi); 
  convolve.B13_aa(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
}

void convolve_B13_ea(void *buffers[], void *cl_arg)
{
  double *datak, *datai, *datao;  size_t ng, mg, lg, ni, mi, li, no, mo, lo; 
  get_data(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo, buffers, cl_arg);

  size_t r = (ng-1)/2;
  size_t yi=0; size_t xi=0;  mi=2*r;  ni=r;
  datai+=(yi*li+xi); 
  convolve.B13_ea(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
}

// ---------------------------------------------------------------------------- 

void convolve_B23_aa(void *buffers[], void *cl_arg)
{
  double *datak, *datai, *datao;  size_t ng, mg, lg, ni, mi, li, no, mo, lo; 
  get_data(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo, buffers, cl_arg);

  size_t r = (ng-1)/2;
  size_t yi=0; size_t xi=ni-2*r;  mi=mi;  ni=2*r;
  datai+=(yi*li+xi); 
  convolve.B23_aa(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
}

void convolve_B23_ea(void *buffers[], void *cl_arg)
{
  double *datak, *datai, *datao;  size_t ng, mg, lg, ni, mi, li, no, mo, lo; 
  get_data(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo, buffers, cl_arg);

  size_t r = (ng-1)/2;
  size_t yi=0; size_t xi=0;  mi=mi;  ni=r; 
  datai+=(yi*li+xi); 
  convolve.B23_ea(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
}

// ---------------------------------------------------------------------------- 

void convolve_B33_aa(void *buffers[], void *cl_arg)
{
  double *datak, *datai, *datao;  size_t ng, mg, lg, ni, mi, li, no, mo, lo; 
  get_data(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo, buffers, cl_arg);

  size_t r = (ng-1)/2;
  size_t yi=mi-2*r; size_t xi=ni-2*r;  mi=2*r;  ni=2*r;
  datai+=(yi*li+xi); 
  convolve.B33_aa(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
}

void convolve_B33_ea(void *buffers[], void *cl_arg)
{
  double *datak, *datai, *datao;  size_t ng, mg, lg, ni, mi, li, no, mo, lo; 
  get_data(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo, buffers, cl_arg);

  size_t r = (ng-1)/2;
  size_t yi=mi-2*r; size_t xi=0;  mi=2*r;  ni=r;
  datai+=(yi*li+xi); 
  convolve.B33_ea(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
}

void convolve_B33_so(void *buffers[], void *cl_arg)
{
  double *datak, *datai, *datao;  size_t ng, mg, lg, ni, mi, li, no, mo, lo; 
  get_data(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo, buffers, cl_arg);

  size_t r = (ng-1)/2;
  size_t yi=0; size_t xi=ni-2*r;  mi=r;  ni=2*r;
  datai+=(yi*li+xi); 
  convolve.B33_so(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
}

void convolve_B33_se(void *buffers[], void *cl_arg)
{
  double *datak, *datai, *datao;  size_t ng, mg, lg, ni, mi, li, no, mo, lo; 
  get_data(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo, buffers, cl_arg);

  size_t r = (ng-1)/2;
  size_t yi=0; size_t xi=0; mi=r; ni=r;
  datai+=(yi*li+xi); 
  convolve.B33_se(datak, mg, ng, lg, datai, mi, ni, li, datao, mo, no, lo);
}

// ---------------------------------------------------------------------------- 

void load_buffers_cpu(void *buffers[], void *cl_arg)
{
  // Ghost cell 
  double *fo = (double*)STARPU_MATRIX_GET_PTR(buffers[0]);
  size_t no = (unsigned)STARPU_MATRIX_GET_NX(buffers[0]);
  size_t mo = (unsigned)STARPU_MATRIX_GET_NY(buffers[0]);
  size_t lo = (unsigned)STARPU_MATRIX_GET_LD(buffers[0]);

  // Matrix
  double *fi = (double*)STARPU_MATRIX_GET_PTR(buffers[1]);
  size_t ni = (unsigned)STARPU_MATRIX_GET_NX(buffers[1]);
  size_t mi = (unsigned)STARPU_MATRIX_GET_NY(buffers[1]);
  size_t li = (unsigned)STARPU_MATRIX_GET_LD(buffers[1]);

  int tile;
  size_t r;
  starpu_codelet_unpack_args(cl_arg, &tile, &r);
  // printf(" tile:%d, r:%d, mo:%ld, no:%ld, lo:%ld, mi:%ld, ni:%ld, li:%ld, \n", tile, r, mo, no, lo, mi, ni,li);
  load_ghost_cells(fi, mi, ni, li, fo, mo, no, lo, tile, r);
}

struct starpu_codelet add2_cl = 
{ .cpu_funcs = {add_2buffers}, .cpu_funcs_name ={"add2"}, .nbuffers = 3, .modes = {STARPU_W, STARPU_R, STARPU_R } };

struct starpu_codelet add4_cl = 
{ .cpu_funcs = {add_4buffers}, .cpu_funcs_name ={"add4"}, .nbuffers = 5, .modes = {STARPU_W, STARPU_R, STARPU_R, STARPU_R, STARPU_R} };

struct starpu_codelet B11_nw_cl = 
{ .cpu_funcs = {convolve_B11_nw},   .cpu_funcs_name ={"convolve_B11_nw"}, .nbuffers = 3, .modes = {STARPU_W, STARPU_R, STARPU_R} };

struct starpu_codelet B11_no_cl = 
{ .cpu_funcs = {convolve_B11_no},   .cpu_funcs_name ={"convolve_B11_no"}, .nbuffers = 3, .modes = {STARPU_W, STARPU_R, STARPU_R} };

struct starpu_codelet B11_we_cl = 
{ .cpu_funcs = {convolve_B11_we},   .cpu_funcs_name ={"convolve_B11_we"}, .nbuffers = 3, .modes = {STARPU_W, STARPU_R, STARPU_R} };

struct starpu_codelet B11_aa_cl = 
{ .cpu_funcs = {convolve_B11_aa},   .cpu_funcs_name ={"convolve_B11_aa"}, .nbuffers = 3, .modes = {STARPU_W, STARPU_R, STARPU_R} };

struct starpu_codelet B21_we_cl = 
{ .cpu_funcs = {convolve_B21_we}, .cpu_funcs_name = {"convolve_B21_we"}, .nbuffers = 3, .modes = {STARPU_W, STARPU_R, STARPU_R} }; 

struct starpu_codelet B21_aa_cl = 
{ .cpu_funcs = {convolve_B21_aa}, .cpu_funcs_name = {"convolve_B21_aa"}, .nbuffers = 3, .modes = {STARPU_W, STARPU_R, STARPU_R} };

struct starpu_codelet B31_we_cl = 
{ .cpu_funcs = {convolve_B31_we}, .cpu_funcs_name = {"convolve_B31_we"},   .nbuffers = 3, .modes = {STARPU_W, STARPU_R, STARPU_R} };

struct starpu_codelet B31_aa_cl = 
{ .cpu_funcs = {convolve_B31_aa}, .cpu_funcs_name = {"convolve_B31_aa"},   .nbuffers = 3, .modes = {STARPU_W, STARPU_R, STARPU_R} };

struct starpu_codelet B31_sw_cl = 
{ .cpu_funcs = {convolve_B31_sw}, .cpu_funcs_name = {"convolve_B31_sw"},   .nbuffers = 3, .modes = {STARPU_W, STARPU_R, STARPU_R} };

struct starpu_codelet B31_so_cl = 
{ .cpu_funcs = {convolve_B31_so}, .cpu_funcs_name = {"convolve_B31_so"},   .nbuffers = 3, .modes = {STARPU_W, STARPU_R, STARPU_R} };

struct starpu_codelet B12_no_cl = 
{ .cpu_funcs = {convolve_B12_no}, .cpu_funcs_name = {"convolve_B12_no"},   .nbuffers = 3, .modes = {STARPU_W, STARPU_R, STARPU_R} };  

struct starpu_codelet B12_aa_cl =                                                                 
{ .cpu_funcs = {convolve_B12_aa}, .cpu_funcs_name = {"convolve_B12_aa"},    .nbuffers = 3, .modes = {STARPU_W, STARPU_R, STARPU_R} };

struct starpu_codelet B32_aa_cl = 
{ .cpu_funcs = {convolve_B32_aa}, .cpu_funcs_name = {"convolve_B32_aa"},   .nbuffers = 3, .modes = {STARPU_W, STARPU_R, STARPU_R} };

struct starpu_codelet B32_so_cl =                                                                
{ .cpu_funcs = {convolve_B32_so}, .cpu_funcs_name = {"convolve_B32_so"},   .nbuffers = 3, .modes = {STARPU_W, STARPU_R, STARPU_R} };

struct starpu_codelet B13_no_cl = 
{ .cpu_funcs = {convolve_B13_no}, .cpu_funcs_name = {"convolve_B13_no"},    .nbuffers = 3, .modes = {STARPU_W, STARPU_R, STARPU_R}};

struct starpu_codelet B13_ne_cl =                                                               
{ .cpu_funcs = {convolve_B13_ne}, .cpu_funcs_name = {"convolve_B13_ne"},    .nbuffers = 3, .modes = {STARPU_W, STARPU_R, STARPU_R}};

struct starpu_codelet B13_aa_cl =                                                               
{ .cpu_funcs = {convolve_B13_aa}, .cpu_funcs_name = {"convolve_B13_aa"},    .nbuffers = 3, .modes = {STARPU_W, STARPU_R, STARPU_R}};

struct starpu_codelet B13_ea_cl = 
{ .cpu_funcs = {convolve_B13_ea}, .cpu_funcs_name = {"convolve_B13_ea"},    .nbuffers = 3, .modes = {STARPU_W, STARPU_R, STARPU_R}};

struct starpu_codelet B23_aa_cl = 
{ .cpu_funcs = {convolve_B23_aa}, .cpu_funcs_name = {"convolve_B23_aa"},   .nbuffers = 3, .modes = {STARPU_W, STARPU_R, STARPU_R} };

struct starpu_codelet B23_ea_cl =                                                               
{ .cpu_funcs = {convolve_B23_ea}, .cpu_funcs_name = {"convolve_B23_ea"},   .nbuffers = 3, .modes = {STARPU_W, STARPU_R, STARPU_R} };

struct starpu_codelet B33_aa_cl = 
{ .cpu_funcs = {convolve_B33_aa}, .cpu_funcs_name = {"convolve_B33_aa"},   .nbuffers = 3, .modes = {STARPU_W, STARPU_R, STARPU_R}};

struct starpu_codelet B33_ea_cl =                                                              
{ .cpu_funcs = {convolve_B33_ea}, .cpu_funcs_name = {"convolve_B33_ea"},   .nbuffers = 3, .modes = {STARPU_W, STARPU_R, STARPU_R}};

struct starpu_codelet B33_so_cl =                                                              
{ .cpu_funcs = {convolve_B33_so}, .cpu_funcs_name = {"convolve_B33_so"},   .nbuffers = 3, .modes = {STARPU_W, STARPU_R, STARPU_R}};

struct starpu_codelet B33_se_cl =                                                                
{ .cpu_funcs = {convolve_B33_se}, .cpu_funcs_name = {"convolve_B33_se"},   .nbuffers = 3, .modes = {STARPU_W, STARPU_R, STARPU_R}};

struct starpu_codelet load_buffers_cl =                                                                
{.cpu_funcs = {load_buffers_cpu}, .cpu_funcs_name = {"load_buffers"}, .nbuffers = 2, .modes = {STARPU_W, STARPU_R}};


#if B22_ONE_TASK

struct starpu_codelet B22_aa_cl = 
{ .cpu_funcs = {convolve_B22_aa}, .cpu_funcs_name = {"convolve_B22_aa"}, .nbuffers = 3, .modes = {STARPU_W, STARPU_R} }; 

#else 
struct starpu_codelet convolution_sub_cl = 
{
  .cpu_funcs = {compute_convolution_cpu_func},
#if HAS_CUDA
  .cuda_funcs = {compute_convolution_gpu_func},
  .cuda_flags = {STARPU_CUDA_ASYNC},
#endif
  .cpu_funcs_name = {"B22_sub"},
  .nbuffers = 3,
  .modes = {STARPU_W, STARPU_R, STARPU_R}
};

#endif

// ============================================================================
// Task insertion
// ============================================================================

  template<typename T>
void starpu_insert_tasks_convolution(
    TestFunction<T> &f,
    const DH_kernel<T> &d_g, DH_input_mem_conv<T> &d_f, DH_ghost_cells<T> &d_gc, DH_output_mem_conv<T> &d_h, 
    size_t ny, size_t nx)
{
  // Convolution of each block
  if(f.nb[NW]!=-1) starpu_task_insert(&B11_nw_cl, STARPU_W, d_h.buf_nw[0], STARPU_R, d_gc.r[NW], STARPU_R, d_g.full, STARPU_NAME, "B11_nw", 0);
  if(f.nb[NO]!=-1) starpu_task_insert(&B11_no_cl, STARPU_W, d_h.buf_nw[1], STARPU_R, d_gc.r[NO], STARPU_R, d_g.full, STARPU_NAME, "B11_no", 0);
  if(f.nb[WE]!=-1) starpu_task_insert(&B11_we_cl, STARPU_W, d_h.buf_nw[2], STARPU_R, d_gc.r[WE], STARPU_R, d_g.full, STARPU_NAME, "B11_we", 0);

  starpu_task_insert(&B11_aa_cl, STARPU_W, d_h.buf_nw[3], STARPU_R, d_f.input_block_conv[NW], STARPU_R, d_g.full, STARPU_NAME, "B11_aa", 0);
  starpu_task_insert(&add4_cl,
      STARPU_W, d_h.block[NW], STARPU_R, d_h.buf_nw[0], STARPU_R, d_h.buf_nw[1], STARPU_R, d_h.buf_nw[2], STARPU_R, d_h.buf_nw[3], 
      STARPU_NAME, "B11", 0);

  // ---------------------------------------------------------------------------- 
  if(f.nb[NO]!=-1)starpu_task_insert(&B12_no_cl, STARPU_W, d_h.buf_no[0], STARPU_R, d_gc.r[NO], STARPU_R, d_g.full, STARPU_NAME, "B12_no", 0);
  starpu_task_insert(&B12_aa_cl, STARPU_W, d_h.buf_no[1], STARPU_R, d_f.input_block_conv[NO], STARPU_R, d_g.full, STARPU_NAME, "B12_aa", 0);

  starpu_task_insert(&add2_cl, 
      STARPU_W, d_h.block[NO], STARPU_R, d_h.buf_no[0], STARPU_R, d_h.buf_no[1],
      STARPU_NAME, "B12", 0);

  // ---------------------------------------------------------------------------- 
  if(f.nb[NO]!=-1)starpu_task_insert(&B13_no_cl, STARPU_W, d_h.buf_ne[0], STARPU_R, d_gc.r[NO], STARPU_R, d_g.full, STARPU_NAME, "B13_no", 0);
  if(f.nb[NE]!=-1)starpu_task_insert(&B13_ne_cl, STARPU_W, d_h.buf_ne[1], STARPU_R, d_gc.r[NE], STARPU_R, d_g.full, STARPU_NAME, "B13_ne", 0);
  if(f.nb[EA]!=-1)starpu_task_insert(&B13_ea_cl, STARPU_W, d_h.buf_ne[2], STARPU_R, d_gc.r[EA], STARPU_R, d_g.full, STARPU_NAME, "B13_ea", 0);
  starpu_task_insert(&B13_aa_cl, STARPU_W, d_h.buf_ne[3], STARPU_R, d_f.input_block_conv[NE], STARPU_R, d_g.full, STARPU_NAME, "B13_aa", 0);

  starpu_task_insert(&add4_cl, 
      STARPU_W, d_h.block[NE], STARPU_R, d_h.buf_ne[0], STARPU_R, d_h.buf_ne[1], STARPU_R, d_h.buf_ne[2], STARPU_R, d_h.buf_ne[3],
      STARPU_NAME, "B13", 0);


  // ---------------------------------------------------------------------------- 
  if(f.nb[WE]!=-1)starpu_task_insert(&B21_we_cl, STARPU_W, d_h.buf_we[0], STARPU_R, d_gc.r[WE], STARPU_R, d_g.full, STARPU_NAME, "B21_we", 0);
  starpu_task_insert(&B21_aa_cl, STARPU_W, d_h.buf_we[1], STARPU_R, d_f.input_block_conv[WE], STARPU_R, d_g.full, STARPU_NAME, "B21_aa", 0);

  starpu_task_insert(&add2_cl,
      STARPU_W, d_h.block[WE], STARPU_R, d_h.buf_we[0], STARPU_R, d_h.buf_we[1],
      STARPU_NAME, "B12", 0);

  // ---------------------------------------------------------------------------- 
#if B22_ONE_TASK
  {
    starpu_task_insert(&B22_aa_cl, STARPU_W, d_h.block[AA], STARPU_R, d_f.input_block_conv[AA], STARPU_R, d_g.full, STARPU_NAME, "B22_aa", 0);
  }
#else
  vector<string> str(d_h.AA_block.size());
  for (size_t i=0; i<ny; i++)
  {
    for (size_t j=0; j<nx; j++)
    {
      size_t idx = i*nx+j;
      str[idx] = "B22_" + std::to_string(idx);
      starpu_task_insert(&convolution_sub_cl, 
          STARPU_W, d_h.AA_block[idx], STARPU_R, d_f.input_AA_block_conv[idx], STARPU_R, d_g.full,
          STARPU_NAME, str[idx].c_str(), 0);
    }
  }
#endif

  // ---------------------------------------------------------------------------- 
  if(f.nb[EA]!=-1)starpu_task_insert(&B23_ea_cl, STARPU_W, d_h.buf_ea[1], STARPU_R, d_gc.r[EA], STARPU_R, d_g.full, STARPU_NAME, "B23_ea", 0);
  starpu_task_insert(&B23_aa_cl, STARPU_W, d_h.buf_ea[0], STARPU_R, d_f.input_block_conv[EA], STARPU_R, d_g.full, STARPU_NAME, "B23_aa", 0);

  starpu_task_insert(&add2_cl,
      STARPU_W, d_h.block[EA], STARPU_R, d_h.buf_ea[0], STARPU_R, d_h.buf_ea[1],
      STARPU_NAME, "B12", 0);

  // ---------------------------------------------------------------------------- 
  if(f.nb[WE]!=-1)starpu_task_insert(&B31_we_cl, STARPU_W, d_h.buf_sw[0], STARPU_R, d_gc.r[WE], STARPU_R, d_g.full, STARPU_NAME, "B31_we", 0);
  if(f.nb[SW]!=-1)starpu_task_insert(&B31_sw_cl, STARPU_W, d_h.buf_sw[1], STARPU_R, d_gc.r[SW], STARPU_R, d_g.full, STARPU_NAME, "B31_sw", 0);
  if(f.nb[SO]!=-1)starpu_task_insert(&B31_so_cl, STARPU_W, d_h.buf_sw[2], STARPU_R, d_gc.r[SO], STARPU_R, d_g.full, STARPU_NAME, "B31_so", 0);
  starpu_task_insert(&B31_aa_cl, STARPU_W, d_h.buf_sw[3], STARPU_R, d_f.input_block_conv[SW], STARPU_R, d_g.full, STARPU_NAME, "B31_aa", 0);

  starpu_task_insert(&add4_cl,
      STARPU_W, d_h.block[SW], STARPU_R, d_h.buf_sw[0], STARPU_R, d_h.buf_sw[1], STARPU_R, d_h.buf_sw[2], STARPU_R, d_h.buf_sw[3],
      STARPU_NAME, "B13", 0);

  // ---------------------------------------------------------------------------- 
  if(f.nb[SO]!=-1)starpu_task_insert(&B32_so_cl, STARPU_W, d_h.buf_so[0], STARPU_R, d_gc.r[SO], STARPU_R, d_g.full, STARPU_NAME, "B32_so", 0);
  starpu_task_insert(&B32_aa_cl, STARPU_W, d_h.buf_so[1], STARPU_R, d_f.input_block_conv[SO], STARPU_R, d_g.full, STARPU_NAME, "B32_aa", 0);

  starpu_task_insert(&add2_cl,
      STARPU_W, d_h.block[SO], STARPU_R, d_h.buf_so[0], STARPU_R, d_h.buf_so[1], 
      STARPU_NAME, "B12", 0);

  // ---------------------------------------------------------------------------- 
  if(f.nb[EA]!=-1)starpu_task_insert(&B33_ea_cl, STARPU_W, d_h.buf_se[0], STARPU_R, d_gc.r[EA], STARPU_R, d_g.full, STARPU_NAME, "B33_ea", 0);
  if(f.nb[SO]!=-1)starpu_task_insert(&B33_so_cl, STARPU_W, d_h.buf_se[1], STARPU_R, d_gc.r[SO], STARPU_R, d_g.full, STARPU_NAME, "B33_so", 0);
  if(f.nb[SE]!=-1)starpu_task_insert(&B33_se_cl, STARPU_W, d_h.buf_se[2], STARPU_R, d_gc.r[SE], STARPU_R, d_g.full, STARPU_NAME, "B33_se", 0);
  starpu_task_insert(&B33_aa_cl, STARPU_W, d_h.buf_se[3], STARPU_R, d_f.input_block_conv[SE], STARPU_R, d_g.full, STARPU_NAME, "B33_aa", 0);

  starpu_task_insert(&add4_cl, 
      STARPU_W, d_h.block[SE], STARPU_R, d_h.buf_se[0], STARPU_R, d_h.buf_se[1], STARPU_R, d_h.buf_se[2], STARPU_R, d_h.buf_se[3], 
      STARPU_NAME, "B13", 0);
}


  template<typename T>
void starpu_insert_tasks_exchange_ghost_cells(
    int my_rank, TestFunction<T> &f, DH_input_mem_conv<T> &d_f, DH_ghost_cells<T> &d_gc, size_t r)
{
#if MPI_ENABLED
  if(f.nb[NW] != -1) starpu_task_insert(&load_buffers_cl, STARPU_VALUE, &NW, sizeof(NW), STARPU_VALUE, &r, sizeof(r), STARPU_W, d_gc.s[NW], STARPU_R, d_f.full, STARPU_NAME, "Load NW", 0);
  if(f.nb[NO] != -1) starpu_task_insert(&load_buffers_cl, STARPU_VALUE, &NO, sizeof(NO), STARPU_VALUE, &r, sizeof(r), STARPU_W, d_gc.s[NO], STARPU_R, d_f.full, STARPU_NAME, "Load NO", 0);
  if(f.nb[NE] != -1) starpu_task_insert(&load_buffers_cl, STARPU_VALUE, &NE, sizeof(NE), STARPU_VALUE, &r, sizeof(r), STARPU_W, d_gc.s[NE], STARPU_R, d_f.full, STARPU_NAME, "Load NE", 0);
  if(f.nb[WE] != -1) starpu_task_insert(&load_buffers_cl, STARPU_VALUE, &WE, sizeof(WE), STARPU_VALUE, &r, sizeof(r), STARPU_W, d_gc.s[WE], STARPU_R, d_f.full, STARPU_NAME, "Load WE", 0);
  if(f.nb[EA] != -1) starpu_task_insert(&load_buffers_cl, STARPU_VALUE, &EA, sizeof(EA), STARPU_VALUE, &r, sizeof(r), STARPU_W, d_gc.s[EA], STARPU_R, d_f.full, STARPU_NAME, "Load EA", 0);
  if(f.nb[SW] != -1) starpu_task_insert(&load_buffers_cl, STARPU_VALUE, &SW, sizeof(SW), STARPU_VALUE, &r, sizeof(r), STARPU_W, d_gc.s[SW], STARPU_R, d_f.full, STARPU_NAME, "Load SW", 0);
  if(f.nb[SO] != -1) starpu_task_insert(&load_buffers_cl, STARPU_VALUE, &SO, sizeof(SO), STARPU_VALUE, &r, sizeof(r), STARPU_W, d_gc.s[SO], STARPU_R, d_f.full, STARPU_NAME, "Load SO", 0);
  if(f.nb[SE] != -1) starpu_task_insert(&load_buffers_cl, STARPU_VALUE, &SE, sizeof(SE), STARPU_VALUE, &r, sizeof(r), STARPU_W, d_gc.s[SE], STARPU_R, d_f.full, STARPU_NAME, "Load SE", 0);

  // Recv
  starpu_mpi_req recv_requests[n_neighbors];
  starpu_mpi_req send_requests[n_neighbors];

  int n_recv_req=0;
  for (size_t i=0; i<n_neighbors; i++)
  {
    int nb = f.nb[i];
    if(nb==-1) continue;
    starpu_mpi_irecv_detached(d_gc.r[i], nb, (n_neighbors-1)-i,  MPI_COMM_WORLD, NULL, NULL); 
    // starpu_mpi_irecv(d_gc.r[i], &recv_requests[n_recv_req++], nb, (n_neighbors-1)-i,  MPI_COMM_WORLD); 
  }

  // Send
  int n_send_req=0;
  for (size_t i=0; i<n_neighbors; i++)
  {
    int nb = f.nb[i];
    if( nb ==-1) continue;
    starpu_mpi_isend_detached(d_gc.s[i], nb, i, MPI_COMM_WORLD,  NULL, NULL); 
    // starpu_mpi_isend(d_gc.s[i], &send_requests[n_send_req++], nb, i, MPI_COMM_WORLD); 
  }

#else

  // Ghost cells
  starpu_task_insert(&load_buffers_cl, STARPU_VALUE, &SE, sizeof(SE), STARPU_VALUE, &r, sizeof(r), STARPU_W, d_gc.r[NW], STARPU_R, d_f.full, STARPU_NAME, "send/recv:NW", 0);
  starpu_task_insert(&load_buffers_cl, STARPU_VALUE, &SO, sizeof(SO), STARPU_VALUE, &r, sizeof(r), STARPU_W, d_gc.r[NO], STARPU_R, d_f.full, STARPU_NAME, "send/recv:NO", 0);
  starpu_task_insert(&load_buffers_cl, STARPU_VALUE, &SW, sizeof(SW), STARPU_VALUE, &r, sizeof(r), STARPU_W, d_gc.r[NE], STARPU_R, d_f.full, STARPU_NAME, "send/recv:NE", 0);
  starpu_task_insert(&load_buffers_cl, STARPU_VALUE, &EA, sizeof(EA), STARPU_VALUE, &r, sizeof(r), STARPU_W, d_gc.r[WE], STARPU_R, d_f.full, STARPU_NAME, "send/recv:NW", 0);
  starpu_task_insert(&load_buffers_cl, STARPU_VALUE, &WE, sizeof(WE), STARPU_VALUE, &r, sizeof(r), STARPU_W, d_gc.r[EA], STARPU_R, d_f.full, STARPU_NAME, "send/recv:NW", 0);
  starpu_task_insert(&load_buffers_cl, STARPU_VALUE, &NE, sizeof(NE), STARPU_VALUE, &r, sizeof(r), STARPU_W, d_gc.r[SW], STARPU_R, d_f.full, STARPU_NAME, "send/recv:NE", 0);
  starpu_task_insert(&load_buffers_cl, STARPU_VALUE, &NO, sizeof(NO), STARPU_VALUE, &r, sizeof(r), STARPU_W, d_gc.r[SO], STARPU_R, d_f.full, STARPU_NAME, "send/recv:NO", 0);
  starpu_task_insert(&load_buffers_cl, STARPU_VALUE, &NW, sizeof(NW), STARPU_VALUE, &r, sizeof(r), STARPU_W, d_gc.r[SE], STARPU_R, d_f.full, STARPU_NAME, "send/recv:NW", 0);

#endif
}

