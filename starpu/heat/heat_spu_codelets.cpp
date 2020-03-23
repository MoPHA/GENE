// ============================================================================ 
// Data handlers
// ============================================================================ 
template<typename T>
class DH_input_mem_heat : public DH_input_mem_conv<T>
{
  public:
    vector<starpu_data_handle_t> block;
    vector<starpu_data_handle_t> AA_block;

    void register_views(TestFunction<T> &f, const Kernel2D<T> &g, size_t msub, size_t nsub, size_t ny, size_t nx)
    {

      int r = g.r;
      size_t m = f.m;
      size_t n = f.n;

      DH_input_mem_conv<T>::register_views(f,g, msub, nsub, ny, nx);

      block.resize(n_blocks);
      {
        for (int iblock=0; iblock<n_neighbors; iblock++)
        {
          size_t yo; size_t xo; size_t mo; size_t no; size_t lo; T* datao;
               if (iblock == NW) { yo = 0   ; xo = 0   ; mo = r     ; no = r     ; lo = f.l ; }
          else if (iblock == NO) { yo = 0   ; xo = r   ; mo = r     ; no = n-2*r ; lo = f.l ; }
          else if (iblock == NE) { yo = 0   ; xo = n-r ; mo = r     ; no = r     ; lo = f.l ; }
          else if (iblock == WE) { yo = r   ; xo = 0   ; mo = m-2*r ; no = r     ; lo = f.l ; }
          else if (iblock == AA) { yo = r   ; xo = r   ; mo = m-2*r ; no = n-2*r ; lo = f.l ; }
          else if (iblock == EA) { yo = r   ; xo = n-r ; mo = m-2*r ; no = r     ; lo = f.l ; }
          else if (iblock == SW) { yo = m-r ; xo = 0   ; mo = r     ; no = r     ; lo = f.l ; }
          else if (iblock == SO) { yo = m-r ; xo = r   ; mo = r     ; no = n-2*r ; lo = f.l ; }
          else if (iblock == SE) { yo = m-r ; xo = n-r ; mo = r     ; no = r     ; lo = f.l ; }
          datao = &f.data[yo*lo+xo];
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
            size_t lo=f.l;
            T* datao=&f.data[yo*lo+xo]; 
            starpu_matrix_data_register(&AA_block[idx], STARPU_MAIN_RAM, (uintptr_t)datao, lo, no, mo, sizeof(datao[0]));
          }
        }
      }
    }

    void unregister_views()
    {
      DH_input_mem_conv<T>::unregister_views();
      for (int i=0; i<AA_block.size(); i++) starpu_data_unregister(AA_block[i]);
      for (int i=0; i<block.size(); i++) starpu_data_unregister(block[i]);
    }
};

// ============================================================================ 
// Codelets and functions
// ============================================================================ 

  template<typename T>
void scale_and_update(
    T* datao, T* datai, size_t m, size_t n, size_t l, 
    T factor
    )
{
  for (int i=0; i<m; i++)
  {
    for (int j=0; j<n; j++)
    {
      datai[i*l+j] *= factor;
      datao[i*l+j] += datai[i*l+j];
    }
  }
}

template<typename T>
void scale_and_update_cpu(void *buffers[], void *cl_arg)
{
  T *fi, *fo;
  size_t mi, ni, li;
  size_t mo, no, lo;

  fo = (T*)STARPU_MATRIX_GET_PTR(buffers[0]);
  no = (unsigned)STARPU_MATRIX_GET_NX(buffers[0]);
  mo = (unsigned)STARPU_MATRIX_GET_NY(buffers[0]);
  lo = (unsigned)STARPU_MATRIX_GET_LD(buffers[0]);

  fi = (T*)STARPU_MATRIX_GET_PTR(buffers[1]);
  ni = (unsigned)STARPU_MATRIX_GET_NX(buffers[1]);
  mi = (unsigned)STARPU_MATRIX_GET_NY(buffers[1]);
  li = (unsigned)STARPU_MATRIX_GET_LD(buffers[1]);

  T factor;
  starpu_codelet_unpack_args(cl_arg, &factor);

  scale_and_update(fo, fi, mo, no, lo, factor);
}

struct starpu_codelet scale_and_update_cl = 
{ .cpu_funcs = {scale_and_update_cpu<double>}, .cpu_funcs_name ={"scale_and_update"}, .nbuffers = 2, .modes = {STARPU_W, STARPU_W} };


