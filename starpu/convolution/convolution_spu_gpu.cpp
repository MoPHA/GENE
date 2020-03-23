#include <starpu.h>
#include "convolution_core.h"

#if 0
extern void compute_convolution_gpu(
    vector< TestFunction<float>* > &subi, 
    TestKernel<float> &g,
    vector< TestFunction<float>* > &subo,
    int M, int N
    );

  template<typename T>
void compute_convolution_cpu(
    vector< TestFunction<T>* > &subi, 
    TestKernel<T> &g,
    vector< TestFunction<T>* > &subo
    )
{
  for (int i=0; i<subi.size(); i++)
    compute_convolution_sub(*subi[i], g, *subo[i]);
} 
#endif

extern void compute_convolution_gpu_func(void *buffers[], void *cl_arg);

void compute_convolution_cpu_func(void *buffers[], void *_args)
{
  float *fo, *fi, *fk;
  size_t no, mo, lo, ni, mi, li, nk, mk, lk;

  fo = (float*)STARPU_MATRIX_GET_PTR(buffers[0]);
  no = (unsigned)STARPU_MATRIX_GET_NX(buffers[0]);
  mo = (unsigned)STARPU_MATRIX_GET_NY(buffers[0]);
  lo = (unsigned)STARPU_MATRIX_GET_LD(buffers[0]);

  fi = (float*)STARPU_MATRIX_GET_PTR(buffers[1]);
  ni = (unsigned)STARPU_MATRIX_GET_NX(buffers[1]);
  mi = (unsigned)STARPU_MATRIX_GET_NY(buffers[1]);
  li = (unsigned)STARPU_MATRIX_GET_LD(buffers[1]);

  fk = (float*)STARPU_MATRIX_GET_PTR(buffers[2]);
  nk = (unsigned)STARPU_MATRIX_GET_NX(buffers[2]);
  mk = (unsigned)STARPU_MATRIX_GET_NY(buffers[2]);
  lk = (unsigned)STARPU_MATRIX_GET_LD(buffers[2]);

  compute_convolution_sub( fk, mk, nk, lk, fi, mi, ni, li, fo, mo, no, lo);
}

  template<typename T>
void create_io_data_handlers(
    TestKernel<T> &g, 
    TestFunction<T> &mati,
    TestFunction<T> &mato,
    vector<starpu_data_handle_t> &dh_smati,
    vector<starpu_data_handle_t> &dh_smato,
    int m_sub, int n_sub, int ms, int ns)
{
  int n_handlers = m_sub*n_sub;
  dh_smati.resize(n_handlers);
  dh_smato.resize(n_handlers);

  int r = g.radius;
  for (int i=0; i<m_sub; i++)
  {
    for (int j=0; j<n_sub; j++)
    {
      int idx = i*n_sub+j;

      int yo=r+i*ms;
      int xo=r+(ns*j);
      int mo=ms; 
      int no=ns; 
      int lo=mato.stride;

      int yi=yo-r;
      int xi=xo-r;
      int mi=mo+2*r;
      int ni=no+2*r;
      int li=mati.stride;

      float* datai=&mati.data[yi*li+xi]; 
      float* datao=&mato.data[yo*lo+xo]; 

      starpu_matrix_data_register(&dh_smati[idx], STARPU_MAIN_RAM, (uintptr_t)datai, li, ni, mi, sizeof(datai[0]));
      starpu_matrix_data_register(&dh_smato[idx], STARPU_MAIN_RAM, (uintptr_t)datao, lo, no, mo, sizeof(datao[0]));
    }
  }
}

struct starpu_codelet convolution_sub_cl = 
{
  .cpu_funcs = {compute_convolution_cpu_func},
  .cuda_funcs = {compute_convolution_gpu_func},
  // .cuda_flags = {STARPU_CUDA_ASYNC},
  .cpu_funcs_name = {"compute_convolution_cpu"},
  .nbuffers = 3,
  .modes = {STARPU_W, STARPU_R, STARPU_R}
};

int main (int argc, char** argv)
{
  int r=2;
  int ms = 1920;
  int ns = 1080;
  int n_sub = 10;
  int m_sub = 10;
  int M = (ms * m_sub +2*r);
  int N = (ns * n_sub +2*r);

  TestFunction<float> mati(M, N, 1);  // input 
  TestFunction<float> mato(M, N, 0);  // output(computed)
  TestKernel<float> g(r,0);           // kernel

  starpu_memory_pin(mati.data, mati.get_mem_size());
  starpu_memory_pin(mato.data, mato.get_mem_size());

  int ret = starpu_init(NULL);

  // Register kernel
  starpu_data_handle_t  hk;
  starpu_matrix_data_register(&hk, STARPU_MAIN_RAM, (uintptr_t)g.data, g.stride, g.x_num, g.y_num, sizeof(g.data[0]));

  starpu_data_handle_t ho;
  starpu_matrix_data_register(&ho, STARPU_MAIN_RAM, (uintptr_t)mato.data, mato.stride, mato.x_num, mato.y_num, sizeof(mato.data[0]));

  // Register local views 
  vector<starpu_data_handle_t> dh_smati;
  vector<starpu_data_handle_t> dh_smato;
  create_io_data_handlers(g, mati, mato, dh_smati, dh_smato, m_sub, n_sub, ms, ns);

  // Create tasks
  for (int i=0; i<dh_smati.size(); i++)
  {
    starpu_task_insert(&convolution_sub_cl, 
        STARPU_VALUE, &M, sizeof(M), STARPU_VALUE, &N, sizeof(N),
        STARPU_W, dh_smato[i], STARPU_R, dh_smati[i], STARPU_R, hk,
        STARPU_NAME, "B33_ea", 0);
  }

  starpu_data_unregister(ho);
  starpu_data_unregister(hk);
  for (int i=0; i<dh_smati.size(); i++) starpu_data_unregister(dh_smati[i]);
  for (int i=0; i<dh_smato.size(); i++) starpu_data_unregister(dh_smato[i]);

  starpu_memory_unpin(&mati, sizeof(mati));
  starpu_memory_unpin(&mato, sizeof(mato));

  printf("Convolution: mat (%d, %d), ke(%d, %d), tasks(%d, %d)=%d, task_size(%d,%d)\n",
      M, N, g.y_num, g.x_num, m_sub, n_sub, m_sub*n_sub, ms, ns);

  printf("Compute reference: CPU serial ...\n");
  TestFunction<float> mato_r(M, N, 0); 

  float *datar = &mato_r.data[r*mato_r.stride+r];
  size_t mr = mato_r.y_num - 2*r;
  size_t nr = mato_r.x_num - 2*r;
  size_t lr = mato_r.stride;

  float *datai = mati.data;
  size_t mi = mati.y_num;
  size_t ni = mati.x_num;
  size_t li = mati.stride;

  float *datak = g.data;
  size_t mk = g.y_num;
  size_t nk = g.x_num;
  size_t lk = g.stride;

  compute_convolution_sub(datak, mk, nk, lk, datai, mi, ni, li, datar, mr, nr, lr);

  datar = mato_r.data;
  mr = mato_r.y_num;
  nr = mato_r.x_num;

  float *datao = mato.data;
  size_t mo = mato.y_num;
  size_t no = mato.x_num;
  size_t lo = mato.stride;

  // // print_matrix(mk, nk, lk, datak);
  // // print_matrix(mi, ni, li, datai);
  // print_matrix(mr, nr, lr, datar);
  // print_matrix(mo, no, lo, datao);

  float error = compute_error(mato_r.data, mato.data, mato.y_num, mato.x_num);
  printf("Abs error: %5.3f\n\n", error);

  starpu_shutdown();
}

