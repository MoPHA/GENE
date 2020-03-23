#define MPI_ENABLED 1
#define TEST_CORRECTNESS   1

#include <starpu_mpi.h>
#include <string.h>
#include <chrono>
#include <vector>

#include "convolution_kernels.h"
#include "convolution_spu_codelets.cpp"
#include "convolution_utilities_mpi.h"

using namespace std;

void parse_input(size_t argc, char **argv,
    size_t n_ranks_y , size_t n_ranks_x ,
    size_t &msup     , size_t &nsup     ,
    size_t &m        , size_t &n        ,
    size_t &msub     , size_t &nsub     ,
    size_t &ny       , size_t &nx       ,
    size_t &r        , int &ncpus)
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
  }

#if B22_ONE_TASK
  nx=1;
  ny=1;
#endif

  if(ny<0) ny=1;

  if (msup != -1) m = msup/n_ranks_y;

  if ( m != -1 ) msub = m/ny;

  m = ny*msub+2*r;
  assert(m-1>r);
  msup = m*n_ranks_y;


  if(nx<0) nx=1;

  if (nsup != -1) n = nsup/n_ranks_x;

  if (n != -1) nsub = n/nx;

  n = nx*nsub+2*r;
  assert(n-1>r);
  nsup = n*n_ranks_x;
}




  template<typename T>
void execute_convolution(
    int argc, char **argv)
{
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  starpu_fxt_autostart_profiling(0);

  // ---------------------------------------------------------------------------- 
  // Init data
  // ---------------------------------------------------------------------------- 

  InitType f_inittype = InitType::CONS;
  InitType h_inittype = InitType::ZERO;

  int my_rank, n_ranks;
  starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
  starpu_mpi_comm_rank(MPI_COMM_WORLD, &my_rank);
  starpu_mpi_comm_size(MPI_COMM_WORLD, &n_ranks);


  if (n_ranks < 4)
  {
    printf("This MPI implementation needs at least 4 processes\n");
    return;
  }

  size_t n_ranks_y = sqrt(n_ranks);
  size_t n_ranks_x = sqrt(n_ranks);

  int mpi_grid[DIMS];
  mpi_grid[Y] = n_ranks_y;
  mpi_grid[X] = n_ranks_x;

  size_t msup, nsup; // Discretization points
  size_t m, n;       // Size of the input matrix in one rank
  size_t ny, nx;        // Submatrices to divide B22
  size_t msub, nsub;
  size_t r;
  int dummy;
  parse_input(argc, argv, n_ranks_y, n_ranks_x, msup, nsup, m, n, msub, nsub, ny, nx, r, dummy);

  // Pointers to global matrix
  TestFunction<T> *hh = NULL;

  // Local input/output
  TestFunction<T> f(m, n, InitType::ZERO);
  TestFunction<T> h(m, n, h_inittype);
  const TestKernel<T> g(r);

  if (my_rank == 0)
  {

    printf("      Global matrix: (%4ld, %4ld) \n",msup, nsup);
    printf("        Rank matrix: (%4ld, %4ld) \n",m, n); 
    printf("      # submatrices: (%4ld, %4ld) %ld\n", ny, nx, ny*nx); 
    printf("B22 sub matrix size: (%4ld, %4ld) \n", msub, nsub); 
    printf("       Kernel size : (%4ld, %4ld) r:%ld\n", g.m, g.n, g.r); 

  }

  find_neighbours_periodic(my_rank, f.nb, mpi_grid);
  f.allocate_buffers_for_ghost_cells(g);


  // ============================================================================ 
  // Create data handlers
  // ============================================================================ 

  DH_kernel<T> d_g;
  d_g.register_views(g);

  DH_input_mem_conv<T> d_f; 
  d_f.register_views(f, g, msub, nsub, ny, nx);

  DH_ghost_cells<T> d_gc;
  d_gc.register_views(f);

  DH_output_mem_conv<T> d_h;
  d_h.register_views(h, g, msub, nsub, ny, nx);

  // ============================================================================ 
  // Distribute data
  // ============================================================================ 

  DH_Global<T> d_ff, d_hh;

  MPI_Status mpi_status;

  if(my_rank == 0)
  {
    TestFunction<T> *ff = NULL;

    // Initialize global matrix
    ff = new TestFunction<T>(msup, nsup, f_inittype);
    hh = new TestFunction<T>(msup, nsup, h_inittype);

    d_ff.register_views(*ff, mpi_grid, my_rank, m, n);
    d_hh.register_views(*hh, mpi_grid, my_rank, m, n);

    starpu_data_acquire(d_ff.block[0], STARPU_R);
    starpu_data_acquire(d_f.full, STARPU_W);

    // Copy
    for (int i=0; i<m; i++)
      for(int j=0; j<n; j++)
        f.data[i*f.l+j] = ff->data[i*ff->l+j];

    starpu_data_release(d_ff.block[0]);
    starpu_data_release(d_f.full);

    // Send corresponding block 
    for (int mpi_rank=1; mpi_rank<n_ranks; mpi_rank++)
    {
      int dest = mpi_rank;
      size_t tag = 0;
      // starpu_mpi_isend_detached(d_ff.block[mpi_rank], dest, -1,  MPI_COMM_WORLD, NULL, NULL); 
      starpu_mpi_send(d_ff.block[mpi_rank], dest, -1,  MPI_COMM_WORLD); 
    }
  }
  else
  {
    int source = 0;
    size_t tag = 0;
    starpu_mpi_recv(d_f.full, source, -1,  MPI_COMM_WORLD, &mpi_status); 
  }

  // printf("[%d] Data received\n", my_rank);

  // starpu_mpi_barrier(MPI_COMM_WORLD);

  //============================================================================ 
  // Insert tasks
  //============================================================================ 

  double start1 = starpu_timing_now();
  starpu_fxt_start_profiling();

  starpu_insert_tasks_exchange_ghost_cells(my_rank, f, d_f, d_gc, r);

  starpu_mpi_barrier(MPI_COMM_WORLD);

  starpu_insert_tasks_convolution(f, d_g, d_f, d_gc, d_h, ny, nx);

  starpu_task_wait_for_all();

  starpu_fxt_stop_profiling();

  double end1 = starpu_timing_now();

  double timing1 = end1 - start1;
  printf("Execution time %2.3f \n", (T)timing1/10e6);

  double start2 = starpu_timing_now();

  //============================================================================ 
  // Collect data
  //============================================================================ 

  starpu_mpi_barrier(MPI_COMM_WORLD);

  // print_by_rank(n_ranks, my_rank, f);

  if(my_rank == 0)
  {
    MPI_Status mpi_status;
    // Recv from all ranks
    for (int mpi_rank=1; mpi_rank<n_ranks; mpi_rank++)
      // starpu_mpi_irecv_detached(d_hh.block[mpi_rank], mpi_rank, 0, MPI_COMM_WORLD, NULL, NULL); 
      starpu_mpi_recv(d_hh.block[mpi_rank], mpi_rank, -2,  MPI_COMM_WORLD, &mpi_status); 
  }
  else
  {
    // Send to root
    // starpu_mpi_isend_detached(d_h.full, 0, 0, MPI_COMM_WORLD, NULL, NULL); 
    starpu_mpi_send(d_h.full, 0, -2,  MPI_COMM_WORLD); 
  }

  starpu_mpi_barrier(MPI_COMM_WORLD);

  // ============================================================================ 
  // Deregister
  // ============================================================================ 
  d_g.unregister_views();
  d_f.unregister_views();
  d_gc.unregister_views(f);
  d_h.unregister_views();

  d_ff.unregister_views();
  d_hh.unregister_views();

#if B22_ONE_TASK
#else
  starpu_memory_unpin(&f, sizeof(f));
  starpu_memory_unpin(&h, sizeof(h));
#endif

  starpu_mpi_barrier(MPI_COMM_WORLD);

  if (my_rank == 0)
  {

    // Copy
    for (int i=0; i<m; i++)
      for(int j=0; j<n; j++)
        hh->data[i*hh->l+j] = h.data[i*h.l+j];

#if TEST_CORRECTNESS
    printf("Convolution serial ... \n");
    std::chrono::steady_clock::time_point tbegin = std::chrono::steady_clock::now();
    TestFunctionConv<T> ff0(msup, nsup, r, f_inittype);
    TestFunction<T>     hh0(msup, nsup,    h_inittype);
    compute_serial_convolution(ff0, g, hh0);

    // print_matrix(ff0.m, ff0.n, ff0.l, ff0.data);     // reference
    // print_matrix(g.m, g.n, g.l, g.data);
    // print_matrix(hh0.m, hh0.n, hh0.l, hh0.data); // reference
    // print_matrix(hh->m, hh->n, hh->l, hh->data); // computed

    double error = compute_error(hh0.data, hh->data, hh0.m, hh0.n);
    printf("----------------------------------------------------------------------------\n");
    printf("Blocked convolution: Abs error: %5.3f\n", error);
    printf("----------------------------------------------------------------------------\n");

    std::chrono::steady_clock::time_point tend = std::chrono::steady_clock::now();
    size_t elapsed_time = chrono::duration_cast<chrono::microseconds>(tend - tbegin).count();
    printf("Total serial convolution: %2.3ld \n", elapsed_time);

#endif //TEST_CORRECTNESS
  }

  starpu_mpi_barrier(MPI_COMM_WORLD);

  double start3 = starpu_timing_now();
  starpu_mpi_shutdown();
  double end3 = starpu_timing_now();
  double timing3 = end3 - start3;
  //printf("Shutdown data: %2.3f \n", (double)timing3/10e6);

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  size_t elapsed_time = chrono::duration_cast<chrono::microseconds>(end - begin).count();
  //printf("Time convolution = %lu [us]\n", elapsed_time);
}


//============================================================================ 
// Main
//============================================================================ 
int main(int argc, char **argv)
{
  execute_convolution<double>(argc, argv);
}

