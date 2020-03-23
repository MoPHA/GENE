#define MPI_ENABLED 1
#define B22_ONE_TASK 0

#include <starpu_mpi.h>
#include <vector>
#include <string>
#include <cmath>
#include "convolution_kernels.h"
#include "convolution_utilities_mpi.h"
#include "convolution_spu_codelets.cpp"
#include "heat_spu_codelets.cpp"


using namespace std;

#define N_ITERS 1250
// #define N_ITERS 20000 
#define EPS 1e-8
const int YMIN=0;     
const int XMIN=0;     
const int YMAX=20;      
const int XMAX=20;

template<typename T>
class Domain1D
{
  public:
    T min; 
    T max;
    size_t n_points;
    Domain1D() : min(0), max(0), n_points(0) {};
    Domain1D(T init, T end, size_t n_points) : min(init), max(end), n_points(n_points) {};
    string to_string(){ return string( "[" + std::to_string(min) + "," + std::to_string(max) + ")"); }
};

  template<typename T>
void solve_unsteady_2D_heat_equation(int argc, char **argv)
{

  int my_rank, n_ranks;
  starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
  starpu_mpi_comm_rank(MPI_COMM_WORLD, &my_rank);
  starpu_mpi_comm_size(MPI_COMM_WORLD, &n_ranks);

  // ============================================================================ 
  // Distribute nodes in a cartesian grid and find neighbours for each node
  // ============================================================================ 
  int mpi_grid[DIMS];
  int n_ranks_y = sqrt(n_ranks); //(size % 2 )? 1: 2;
  int n_ranks_x = sqrt(n_ranks); //(size / n_ranks_y);
  mpi_grid[Y] = n_ranks_y;
  mpi_grid[X] = n_ranks_x;

  int mpi_coord[DIMS];
  mpi_coord[Y] = my_rank / n_ranks_x;
  mpi_coord[X] = my_rank % n_ranks_x;

  // ============================================================================ 
  // Define global domain discretization 
  // ============================================================================ 
  static Convolution<T> convolve;
  vector<Domain1D<T>> domain_g; // Domain global
  domain_g.push_back(Domain1D<T>(YMIN, YMAX, YMAX-YMIN));
  domain_g.push_back(Domain1D<T>(XMIN, XMAX, XMAX-XMIN));

  if ( domain_g[Y].n_points*domain_g[X].n_points < my_rank )
  {
    if(my_rank == 0)
      printf("There are more processes than discrete points. Terminate program\n");
    // MPI_Finalize();
    exit(0);
  }

  if ( my_rank  == 0)
  {
    printf("Domain: y:%s x:%s \n", domain_g[Y].to_string().c_str(), domain_g[X].to_string().c_str());
    printf("Use %d proc. in a cart. grid y=%d, x=%d\n", n_ranks, mpi_grid[Y], mpi_grid[X]);
  }

  // ============================================================================ 
  // Define subdomain for this mpi rank
  // ============================================================================ 

  vector<Domain1D<int>> domain_l(DIMS);
  for (int i = 0; i<DIMS; i++)
  {
    int len = (domain_g[i].n_points) / mpi_grid[i];
    int rem = (domain_g[i].n_points) - (mpi_grid[i] * len);

    size_t min, max;
    min = mpi_coord[i]*len;
    max = min + len;

    if(my_rank > 0)
    {
      if(mpi_coord[i] < rem)
      {
        min += mpi_coord[i];
        max += mpi_coord[i];
      }
      else
      {
        min += rem;
        max += rem;
      }
    }

    domain_l[i].min = min;
    domain_l[i].max = max;
    domain_l[i].n_points = max - min;
  }
  printf("[%d] (%d, %d), Domain: y:%s x:%s \n", my_rank, mpi_coord[Y], mpi_coord[X],  domain_l[Y].to_string().c_str(), domain_l[X].to_string().c_str());

  // Initialize
  T dx = 1.0/(XMAX-1);
  T dy = 1.0/(YMAX-1);
  T dx2 = dx*dx;
  T dy2 = dy*dy;
  T dx2i = 1.0 / (6*dx2);
  T dy2i = 1.0 / (6*dy2);
  T dt = min(dx2, dy2)/4.0;
  T k=1.0;
  T factor = k*dt*dx2i;

  // ============================================================================ 
  // Initialize 
  // ============================================================================ 
  TestFunction<T>  u(domain_l[Y].n_points, domain_l[X].n_points, InitType::ZERO);
  TestFunction<T> du(domain_l[Y].n_points, domain_l[X].n_points, InitType::ZERO);
  LaplaceOperator9<T> g;

  if(domain_l[Y].min == domain_g[Y].min)
  {
    size_t i=0;
    for(size_t j=0; j<domain_l[X].n_points; j++)
    {
      u.data[i*u.l+j] = (domain_l[X].min+j)*dx; //south
    }
  }

  if(domain_l[Y].max == domain_g[Y].max)
  {
    size_t i = domain_l[Y].n_points-1; 
    for(size_t j=0; j<domain_l[X].n_points; j++)
    {
      u.data[i*u.l+j] = (domain_l[X].min+j)*dx; //north
    }
  }

  if(domain_l[X].min == domain_g[X].min)
  {
    size_t j = 0; 
    for(size_t i=0; i<domain_l[Y].n_points; i++)
    {
      u.data[i*u.l+j] = 0; //west
    }
  }

  if(domain_l[X].max == domain_g[X].max)
  {
    size_t j = domain_l[X].n_points-1; 
    for(size_t i=0; i<domain_l[Y].n_points; i++)
    {
      u.data[i*u.l+j] = 1.0; //east
    }
  }

  // Find neighbors and allocate ghost cells
  find_neighbours(my_rank, u.nb, mpi_grid);
  u.allocate_buffers_for_ghost_cells(g);

  size_t yy = (u.nb[NO]==-1)?1 :0;
  size_t xx = (u.nb[WE]==-1)?1 :0;
  size_t mm = (u.nb[SO]==-1)?(u.m-1): u.m;
  size_t nn = (u.nb[EA]==-1)?(u.n-1): u.n;

  if (my_rank == 0)
  {
    printf("Unsteady 2D Heat equation solved by FTCD \n");
    printf("dx = %5.3e, dy = %5.3e, dt = %5.3e, eps = %5.3e, factor = %5.3e,\n", dx, dy, dt, EPS, factor);
    printf("============================================================================\n");
  }

  size_t ny = 2;
  size_t nx = 2;
  size_t msub = 4;
  size_t nsub = 4;

  DH_kernel<T> d_g;
  d_g.register_views(g);

  DH_input_mem_heat<T> d_f; 
  d_f.register_views(u, g, msub, nsub, ny, nx);

  DH_ghost_cells<T> d_gc;
  d_gc.register_views(u);

  DH_output_mem_conv<T> d_h;
  d_h.register_views(du, g, msub, nsub, ny, nx);


  // ============================================================================ 
  //  Solve with FTCD
  // ============================================================================ 

  T exec_time = 0; 
  T comm_time = 0;
  T criterion_time = 0;
  int stride=100;
  bool done = false;

  int iter;

  starpu_mpi_barrier(MPI_COMM_WORLD);

  exec_time -= MPI_Wtime();

  for (iter=1; iter<N_ITERS; iter++)
  {

    starpu_iteration_push(iter);

    // ============================================================================  
    // Exchange ghost cells
    // ============================================================================  
    starpu_insert_tasks_exchange_ghost_cells(my_rank, u, d_f, d_gc, 1);

    // ============================================================================  
    // Compute Laplacian
    // ============================================================================  
    {
      if(yy==0 || xx==0)
      {
        // Convolution of each block
        if(u.nb[NW]!=-1) starpu_task_insert(&B11_nw_cl, STARPU_W, d_h.buf_nw[0], STARPU_R, d_gc.r[NW], STARPU_R, d_g.full, STARPU_NAME, "B11_nw", 0);
        if(u.nb[NO]!=-1) starpu_task_insert(&B11_no_cl, STARPU_W, d_h.buf_nw[1], STARPU_R, d_gc.r[NO], STARPU_R, d_g.full, STARPU_NAME, "B11_no", 0);
        if(u.nb[WE]!=-1) starpu_task_insert(&B11_we_cl, STARPU_W, d_h.buf_nw[2], STARPU_R, d_gc.r[WE], STARPU_R, d_g.full, STARPU_NAME, "B11_we", 0);

        starpu_task_insert(&B11_aa_cl, STARPU_W, d_h.buf_nw[3], STARPU_R, d_f.input_block_conv[NW], STARPU_R, d_g.full, STARPU_NAME, "B11_aa", 0);
        starpu_task_insert(&add4_cl,
            STARPU_W, d_h.block[NW], STARPU_R, d_h.buf_nw[0], STARPU_R, d_h.buf_nw[1], STARPU_R, d_h.buf_nw[2], STARPU_R, d_h.buf_nw[3], 
            STARPU_NAME, "B11", 0);
      }

      if(yy==0)
      {
        if(u.nb[NO]!=-1)starpu_task_insert(&B12_no_cl, STARPU_W, d_h.buf_no[0], STARPU_R, d_gc.r[NO], STARPU_R, d_g.full, STARPU_NAME, "B12_no", 0);
        starpu_task_insert(&B12_aa_cl, STARPU_W, d_h.buf_no[1], STARPU_R, d_f.input_block_conv[NO], STARPU_R, d_g.full, STARPU_NAME, "B12_aa", 0);

        starpu_task_insert(&add2_cl, 
            STARPU_W, d_h.block[NO], STARPU_R, d_h.buf_no[0], STARPU_R, d_h.buf_no[1],
            STARPU_NAME, "B12", 0);
      }

      if(yy==0 || nn==u.n)
      {
        if(u.nb[NO]!=-1)starpu_task_insert(&B13_no_cl, STARPU_W, d_h.buf_ne[0], STARPU_R, d_gc.r[NO], STARPU_R, d_g.full, STARPU_NAME, "B13_no", 0);
        if(u.nb[NE]!=-1)starpu_task_insert(&B13_ne_cl, STARPU_W, d_h.buf_ne[1], STARPU_R, d_gc.r[NE], STARPU_R, d_g.full, STARPU_NAME, "B13_ne", 0);
        if(u.nb[EA]!=-1)starpu_task_insert(&B13_ea_cl, STARPU_W, d_h.buf_ne[2], STARPU_R, d_gc.r[EA], STARPU_R, d_g.full, STARPU_NAME, "B13_ea", 0);
        starpu_task_insert(&B13_aa_cl, STARPU_W, d_h.buf_ne[3], STARPU_R, d_f.input_block_conv[NE], STARPU_R, d_g.full, STARPU_NAME, "B13_aa", 0);

        starpu_task_insert(&add4_cl, 
            STARPU_W, d_h.block[NE], STARPU_R, d_h.buf_ne[0], STARPU_R, d_h.buf_ne[1], STARPU_R, d_h.buf_ne[2], STARPU_R, d_h.buf_ne[3],
            STARPU_NAME, "B13", 0);
      }


      if(xx == 0)
      {
        if(u.nb[WE]!=-1)starpu_task_insert(&B21_we_cl, STARPU_W, d_h.buf_we[0], STARPU_R, d_gc.r[WE], STARPU_R, d_g.full, STARPU_NAME, "B21_we", 0);
        starpu_task_insert(&B21_aa_cl, STARPU_W, d_h.buf_we[1], STARPU_R, d_f.input_block_conv[WE], STARPU_R, d_g.full, STARPU_NAME, "B21_aa", 0);

        starpu_task_insert(&add2_cl,
            STARPU_W, d_h.block[WE], STARPU_R, d_h.buf_we[0], STARPU_R, d_h.buf_we[1],
            STARPU_NAME, "B12", 0);
      }

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

      if( nn == u.n)
      {
        if(u.nb[EA]!=-1)starpu_task_insert(&B23_ea_cl, STARPU_W, d_h.buf_ea[1], STARPU_R, d_gc.r[EA], STARPU_R, d_g.full, STARPU_NAME, "B23_ea", 0);
        starpu_task_insert(&B23_aa_cl, STARPU_W, d_h.buf_ea[0], STARPU_R, d_f.input_block_conv[EA], STARPU_R, d_g.full, STARPU_NAME, "B23_aa", 0);

        starpu_task_insert(&add2_cl,
            STARPU_W, d_h.block[EA], STARPU_R, d_h.buf_ea[0], STARPU_R, d_h.buf_ea[1],
            STARPU_NAME, "B12", 0);

      }

      if( mm==u.m || xx==0 )
      {
        // ---------------------------------------------------------------------------- 
        if(u.nb[WE]!=-1)starpu_task_insert(&B31_we_cl, STARPU_W, d_h.buf_sw[0], STARPU_R, d_gc.r[WE], STARPU_R, d_g.full, STARPU_NAME, "B31_we", 0);
        if(u.nb[SW]!=-1)starpu_task_insert(&B31_sw_cl, STARPU_W, d_h.buf_sw[1], STARPU_R, d_gc.r[SW], STARPU_R, d_g.full, STARPU_NAME, "B31_sw", 0);
        if(u.nb[SO]!=-1)starpu_task_insert(&B31_so_cl, STARPU_W, d_h.buf_sw[2], STARPU_R, d_gc.r[SO], STARPU_R, d_g.full, STARPU_NAME, "B31_so", 0);
        starpu_task_insert(&B31_aa_cl, STARPU_W, d_h.buf_sw[3], STARPU_R, d_f.input_block_conv[SW], STARPU_R, d_g.full, STARPU_NAME, "B31_aa", 0);

        starpu_task_insert(&add4_cl,
            STARPU_W, d_h.block[SW], STARPU_R, d_h.buf_sw[0], STARPU_R, d_h.buf_sw[1], STARPU_R, d_h.buf_sw[2], STARPU_R, d_h.buf_sw[3],
            STARPU_NAME, "B13", 0);
      }

      if( mm==u.m)
      {
        // ---------------------------------------------------------------------------- 
        if(u.nb[SO]!=-1)starpu_task_insert(&B32_so_cl, STARPU_W, d_h.buf_so[0], STARPU_R, d_gc.r[SO], STARPU_R, d_g.full, STARPU_NAME, "B32_so", 0);
        starpu_task_insert(&B32_aa_cl, STARPU_W, d_h.buf_so[1], STARPU_R, d_f.input_block_conv[SO], STARPU_R, d_g.full, STARPU_NAME, "B32_aa", 0);

        starpu_task_insert(&add2_cl,
            STARPU_W, d_h.block[SO], STARPU_R, d_h.buf_so[0], STARPU_R, d_h.buf_so[1], 
            STARPU_NAME, "B12", 0);
      }

      if( mm==u.m || nn==u.n)
      {
        // ---------------------------------------------------------------------------- 
        if(u.nb[EA]!=-1)starpu_task_insert(&B33_ea_cl, STARPU_W, d_h.buf_se[0], STARPU_R, d_gc.r[EA], STARPU_R, d_g.full, STARPU_NAME, "B33_ea", 0);
        if(u.nb[SO]!=-1)starpu_task_insert(&B33_so_cl, STARPU_W, d_h.buf_se[1], STARPU_R, d_gc.r[SO], STARPU_R, d_g.full, STARPU_NAME, "B33_so", 0);
        if(u.nb[SE]!=-1)starpu_task_insert(&B33_se_cl, STARPU_W, d_h.buf_se[2], STARPU_R, d_gc.r[SE], STARPU_R, d_g.full, STARPU_NAME, "B33_se", 0);
        starpu_task_insert(&B33_aa_cl, STARPU_W, d_h.buf_se[3], STARPU_R, d_f.input_block_conv[SE], STARPU_R, d_g.full, STARPU_NAME, "B33_aa", 0);

        starpu_task_insert(&add4_cl, 
            STARPU_W, d_h.block[SE], STARPU_R, d_h.buf_se[0], STARPU_R, d_h.buf_se[1], STARPU_R, d_h.buf_se[2], STARPU_R, d_h.buf_se[3], 
            STARPU_NAME, "B13", 0);
      }
    }

    // starpu_task_wait_for_all();
    // print_by_rank(n_ranks, my_rank, du);


    // ============================================================================  
    // Scale and update
    // ============================================================================  
    {
      for (int i=0; i<n_neighbors; i++)
      {
        if(u.nb[i]!=-1)
          starpu_task_insert(&scale_and_update_cl,
              STARPU_VALUE, &factor, sizeof(factor),
              STARPU_W, d_f.block[i],
              STARPU_W, d_h.block[i], STARPU_NAME, "scale&update", 0);
      }
      // starpu_task_wait_for_all();

      // Insert scaling and update tasks
      for (int iblock=0; iblock<d_h.AA_block.size(); iblock++)
      {
        starpu_task_insert(&scale_and_update_cl,
            STARPU_VALUE, &factor, sizeof(factor),
            STARPU_W, d_f.AA_block[iblock],
            STARPU_W, d_h.AA_block[iblock], STARPU_NAME, "scale&update", 0);
      }
    }
    starpu_task_wait_for_all();

#if 1
    // Check convergence every 'stride' times
    if ((iter % stride) == 0)
    {
      starpu_data_acquire(d_h.full, STARPU_R);
      criterion_time -= MPI_Wtime();

      double du_max = 0.0;
      for (int i=0; i<du.m; i++)
      {
        for (int j=0; j<du.n; j++)
        {
          if(du_max < du.data[i+du.l+j])
            du_max = du.data[i+du.l+j];
        }
      }

      T val = du_max;
      MPI_Allreduce(&val, &du_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      printf("-------------- [%d] %f, %u\n",my_rank, du_max, iter);
      criterion_time += MPI_Wtime();

      if (du_max < EPS ) done = true;

      starpu_data_release(d_h.full);
    }
#endif

    starpu_iteration_pop();

    if(done) break; 

  }

  exec_time += MPI_Wtime();

  // print_matrix(u.m, u.n, u.l, u.data);

  d_g.unregister_views();
  d_f.unregister_views();
  d_gc.unregister_views(u);
  d_h.unregister_views();

  if (my_rank==0)
  {
    printf("Iterations: %d\n", iter);
    printf("Total execution: %8.3f [ms]\n", exec_time*1000 );
    printf("Total  MPI comm: %8.3f [ms]\n", comm_time*1000);
    printf("MPI_Allreduce  : %8.3f [ms]\n", criterion_time*1000);
    printf("Check conv.    : %3d iter\n", stride);
  }

  // starpu_mpi_barrier(MPI_COMM_WORLD);
  //print_by_rank(n_ranks, my_rank, u);
  starpu_mpi_shutdown();
  return;
}


int main(int argc, char **argv)
{
  solve_unsteady_2D_heat_equation<double>(argc, argv);
}

