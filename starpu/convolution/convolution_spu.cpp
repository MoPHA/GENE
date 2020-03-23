#define B22_ONE_TASK 0
#define HAS_CUDA     0

#define TEST_CORRECTNESS   1

#include <string.h>
#include <chrono>
#include <vector>

#include "convolution_kernels.h"
#include "convolution_spu_codelets.cpp"

using namespace std;

const int n_iters = 1;

void parse_input(size_t argc, char **argv,
    size_t n_ranks_y , size_t n_ranks_x ,
    size_t &msup     , size_t &nsup     ,
    size_t &m        , size_t &n        ,
    size_t &msub     , size_t &nsub     ,
    size_t &ny       , size_t &nx       ,
    size_t &r        , int &ncpus, 
    int &test_correctness
    )
{

  if(argc<2)
  {
    // **************************************************************************** 
    // Input param. description
    // **************************************************************************** 
    // The global matrix is of dimensions (msup, nsup).
    // The global matrix is split in mpi-rank matrices
    // The size of each mpi-rank matrix is (m,n)
    // Each mpi-rank matrix is divided into 9 sections:
    // NW, NO, NE, WE, AA, EA, SW, SO, SE.
    // The matrix AA is split into submatrices: nx along n, and ny along m
    // and the dimension of each sumatrix is (msub, nsub).
    //
    // Parameters list:
    // msup, nsup, m, n, msub, nsub, ny, nx, r
    //
    // where r is the radius of the convolution kernel

    msup=-1; nsup=-1; // Size of the matrix spread across the ranks
    m=-1; n=-1;       // Size of the matrix in one rank, without padding
    msub = 3;         // Number of submatrices to split B22
    nsub = 5; 
    ny = 2;           // Size of on submatrices which divide B22
    nx = 2;
    r = 1;            // Radius of the kernel
    ncpus=-1;         //Use all cpus
    test_correctness = 1;
  }
  else
  {
    // printf("Read\n");
    msup = atoi(argv[1]);
    nsup = atoi(argv[2]);
    m = atoi(argv[3]);
    n = atoi(argv[4]);
    msub = atoi(argv[5]);
    nsub = atoi(argv[6]);
    ny = atoi(argv[7]);
    nx = atoi(argv[8]);
    r = atoi(argv[9]);
    ncpus = atoi(argv[10]);
    test_correctness = atoi(argv[11]);
  }

#if B22_ONE_TASK
  nx=1;
  ny=1;
#endif

  if(ny<0) ny=1;
  if (msup != -1) m = (msup-2*r)/n_ranks_y;
  if ( m != -1 ) msub = m/ny;

  m = ny*msub+2*r;
  assert(m-1>r);
  msup = m*n_ranks_y;


  if(nx<0) nx=1;
  if (nsup != -1) n = (nsup-2*r)/n_ranks_x;
  if (n != -1) nsub = n/nx;

  n = nx*nsub+2*r;
  assert(n-1>r);
  nsup = n*n_ranks_x;
}


  template<typename T>
void execute_convolution(
    size_t n_ranks_y , size_t n_ranks_x ,
    size_t &msup     , size_t &nsup     ,
    size_t &m        , size_t &n        ,
    size_t &msub     , size_t &nsub     ,
    size_t &ny       , size_t &nx       ,
    size_t &r        , int &ncpus,
    int test_correctness,
    FILE *file)
{

  r = 1;
  msup = 128*64 + 2*r;
  nsup = msup;
  n_ranks_y = 1; 
  n_ranks_x = 1;
  m = msup;
  n = nsup; 
  ncpus = -1;
  test_correctness = 0;

  // ---------------------------------------------------------------------------- 
  // Init data
  // ---------------------------------------------------------------------------- 

  InitType f_inittype = InitType::RAND;
  InitType h_inittype = InitType::ZERO;

  const TestKernel<T> g(r);
  TestFunction<T> f(m, n, f_inittype);
  TestFunction<T> h(m, n, h_inittype);

  f.nb = vector<int>(n_neighbors, 0);
  f.nb[4]=-1;
  // print_matrix(3,3,3,&f.nb[0]);
  f.allocate_buffers_for_ghost_cells(g);
  f.load_ghost_cells_to_send(g);

  for (int i=4; i<5; i++)
  {

    ny = pow(2,i);
    nx = ny;

    printf("\n[%ld] ============================================================================ \n", ny );

    if(ny<0) ny=1;
    if (msup != -1) m = (msup-2*r)/n_ranks_y;
    if ( m != -1 ) msub = m/ny;
    m = ny*msub+2*r;
    assert(m-1>r);
    msup = m*n_ranks_y;

    if(nx<0) nx=1;
    if (nsup != -1) n = (nsup-2*r)/n_ranks_x;
    if (n != -1) nsub = n/nx;
    n = nx*nsub+2*r;
    assert(n-1>r);
    nsup = n*n_ranks_x;

    size_t sum=0;
    for (int iter=0; iter<n_iters; iter++)
    {
      if(ny<0) ny=1;
      if (msup != -1) m = (msup-2*r)/n_ranks_y;
      if ( m != -1 ) msub = m/ny;

      m = ny*msub+2*r;
      assert(m-1>r);
      msup = m*n_ranks_y;


      if(nx<0) nx=1;
      if (nsup != -1) n = (nsup-2*r)/n_ranks_x;
      if (n != -1) nsub = n/nx;

      n = nx*nsub+2*r;
      assert(n-1>r);
      nsup = n*n_ranks_x;

      std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

      struct starpu_conf conf;
      starpu_conf_init(&conf);
      conf.ncpus = ncpus;
      printf("\nSTARPU_NCPUS= %d\n", conf.ncpus);

      int ret = starpu_init (&conf);


      printf(" Global matrix: (%4ld, %4ld) \n",msup, nsup);
      printf("   Rank matrix: (%4ld, %4ld) \n",m, n); 
      printf(" # submatrices: (%4ld, %4ld) %ld\n", ny, nx, ny*nx); 
      printf("B22 sub matrix: (%4ld, %4ld) \n", msub, nsub); 
      printf("  Kernel size : (%4ld, %4ld) r:%ld\n", g.m, g.n, g.r); 


      // ============================================================================ 
      // Create data handlers
      // ============================================================================ 

      double start0 = starpu_timing_now();

      DH_kernel<T> d_g;
      d_g.register_views(g);

      DH_input_mem_conv<T> d_f; 
      d_f.register_views(f, g, msub, nsub, ny, nx);

      DH_ghost_cells<T> d_gc;
      d_gc.register_views(f);

      DH_output_mem_conv<T> d_h;
      d_h.register_views(h, g, msub, nsub, ny, nx);

      // ---------------------------------------------------------------------------- 
      // Output
      // ---------------------------------------------------------------------------- 

      double end0 = starpu_timing_now();
      double timing0 = end0 - start0;
      printf("Register data: %2.3f s \n", (T)timing0/10e6);

      //============================================================================ 
      // Insert tasks
      //============================================================================ 

      double start1 = starpu_timing_now();
      starpu_fxt_start_profiling();

      starpu_insert_tasks_exchange_ghost_cells(1, f, d_f,d_gc, r);

      starpu_insert_tasks_convolution(f, d_g, d_f, d_gc, d_h, ny, nx);

      starpu_task_wait_for_all();

      starpu_fxt_stop_profiling();

      double end1 = starpu_timing_now();

      double timing1 = end1 - start1;
      printf("Execution time %2.3f \n", (T)timing1/10e6);

      double start2 = starpu_timing_now();

      // ============================================================================ 
      // Deregister
      // ============================================================================ 

      d_g.unregister_views();
      d_f.unregister_views();
      d_gc.unregister_views(f);
      d_h.unregister_views();

#if B22_ONE_TASK
#else
      starpu_memory_unpin(&f, sizeof(f));
      starpu_memory_unpin(&h, sizeof(h));
#endif

      if(test_correctness)
      { 
        TestFunction<T> *h0 = &h;
#if TEST_CORRECTNESS
        std::chrono::steady_clock::time_point tbegin = std::chrono::steady_clock::now();

        // TestFunctionConv<T> ff(msup, nsup, r, f_inittype);
        TestFunctionConv<T> ff(f, r);
        TestFunction<T>     hh(msup, nsup,    h_inittype);
        compute_serial_convolution(ff, g, hh);

        // print_matrix(ff.m, ff.n, ff.l, ff.data);     // reference
        // print_matrix(f.m, f.n, f.l, f.data);     // reference
        // print_matrix(g.m, g.n, g.l, g.data);
        // print_matrix(hh.m, hh.n, hh.l, hh.data); // reference
        // print_matrix(h0->m, h0->n, h0->l, h0->data); // computed

        double error = compute_error(hh.data, h0->data, hh.m, hh.n);
        printf("----------------------------------------------------------------------------\n");
        printf("Blocked convolution: Abs error: %5.3f\n", error);
        printf("----------------------------------------------------------------------------\n");

        std::chrono::steady_clock::time_point tend = std::chrono::steady_clock::now();
        size_t elapsed_time = chrono::duration_cast<chrono::microseconds>(tend - tbegin).count();
        printf("Total serial %2.3ld \n", elapsed_time);

#endif //TEST_CORRECTNESS
      }

      double start3 = starpu_timing_now();
      starpu_shutdown();
      double end3 = starpu_timing_now();
      double timing3 = end3 - start3;
      printf("Shutdown data: %2.3f \n", (T)timing3/10e6);

      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
      size_t elapsed_time = chrono::duration_cast<chrono::microseconds>(end - begin).count();
      printf("Time convolution = %lu [us]\n", elapsed_time);
      sum += elapsed_time;
      fflush(file);
    }
    printf("Average: = %lu [us]\n", sum/n_iters);
    fflush(file);
  }
}


//============================================================================ 
// Main
//============================================================================ 
int main(int argc, char **argv)
{
  starpu_fxt_autostart_profiling(0);

  FILE *file = fopen("out_convolution_spu.txt", "w");

  size_t n_ranks_y=1;
  size_t n_ranks_x=1;
  size_t msup, nsup; // Discretization points
  size_t m, n;       // Size of the input matrix in one rank
  size_t ny, nx;        // Submatrices to divide B22
  size_t msub, nsub;
  size_t r;
  int ncpus;
  int test_correctness;
  if(0)
  {
    parse_input(argc, argv, n_ranks_y, n_ranks_x,  msup, nsup, m, n, msub, nsub, ny, nx, r, ncpus, test_correctness);
    execute_convolution<double>(n_ranks_y, n_ranks_x,  msup, nsup, m, n, msub, nsub, ny, nx, r, ncpus, test_correctness, file);
  }

  execute_convolution<double>(n_ranks_y, n_ranks_x,  msup, nsup, m, n, msub, nsub, ny, nx, r, ncpus, test_correctness, file);
  fclose(file);
}

