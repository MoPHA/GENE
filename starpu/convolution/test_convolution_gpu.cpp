#include "convolution_core.h"

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
  printf("[CPU]: Compute convolution ... \n");
  for (int i=0; i<subi.size(); i++)
    compute_convolution_sub(*subi[i], g, *subo[i]);
} 


int main (int argc, char** argv)
{
  int r=2;
  int ms = 2000;
  int ns = 2000;
  int n_sub = 10;
  int m_sub = 10;
  int M = (ms * m_sub);
  int N = (ms * m_sub);

  TestFunctionConv<float> mati(M, N, r);  // input 
  TestFunction<float> mato(M, N, -1);     // output(computed)
  TestFunction<float> mato_r(M, N, -1);   // output(reference)
  TestKernel<float> g(r,0);               // kernel

  printf("Convolution: mat (%d, %d), ke(%d, %d), tasks(%d, %d)=%d, task_size(%d,%d)\n",
      M, N, g.y_num, g.x_num, m_sub, n_sub, m_sub*n_sub, ms, ns);

  // Sub matrices
  vector< TestFunction<float>* > subi(n_sub * m_sub , NULL);
  vector< TestFunction<float>* > subo(n_sub * m_sub , NULL);

  size_t mati_mem = mati.get_mem_size();
  size_t mato_mem = mato.get_mem_size();
  size_t mato_r_mem = mato_r.get_mem_size();
  size_t ke_mem = g.get_mem_size();
  size_t subi_mem, subo_mem;

  printf("Compute reference: CPU serial ...\n");
  compute_serial_convolution(mati, g, mato_r);

  if(1)
  {
    // Decompose large conv_matrix into smaller conv_matrices
    input_matrix_to_submatrices(g, mati, subi,subo, subi_mem, subo_mem, m_sub, n_sub, ms, ns);

    char buf[32];
    pretty_bytes(buf, ke_mem);     printf("ke     :%s\n", buf);
    pretty_bytes(buf, mati_mem);   printf("mati   :%s\n", buf);
    pretty_bytes(buf, mato_mem);   printf("mato   :%s\n", buf);
    pretty_bytes(buf, mato_r_mem); printf("mato_r :%s\n", buf);
    pretty_bytes(buf, subi_mem);   printf("subi   :%s\n", buf);
    pretty_bytes(buf, subo_mem);   printf("subo   :%s\n", buf);

    compute_convolution_cpu(subi, g, subo);
    output_submatrices_into_matrix(subo, mato, m_sub, n_sub, ms, ns);

    float error = compute_error(mato_r.data, mato.data, mato.y_num, mato.x_num);
    printf("[CPU]: Abs error: %5.3f\n\n", error);

    for (int i=0; i<subi.size(); i++)
    {
      if(subi[i]){ delete subi[i]; subi[i]=NULL;}
      if(subo[i]){ delete subo[i]; subo[i]=NULL;}
    }
  }

  if(1)
  {
    // Decompose large conv_matrix into smaller conv_matrices
    input_matrix_to_submatrices(g, mati, subi,subo, subi_mem, subo_mem, m_sub, n_sub, ms, ns);

    compute_convolution_gpu(subi, g, subo, M, N);

    output_submatrices_into_matrix(subo, mato, m_sub, n_sub, ms, ns);

    float error = compute_error(mato_r.data, mato.data, mato.y_num, mato.x_num);
    printf("[GPU]: Abs error: %5.3f\n\n", error);

    for (int i=0; i<subi.size(); i++)
    {
      if(subi[i]){ delete subi[i]; subi[i]=NULL;}
      if(subo[i]){ delete subo[i]; subo[i]=NULL;}
    }
  }
}

