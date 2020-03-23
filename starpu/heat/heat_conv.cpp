#include "../convolution/convolution_core.h" 
#include "../convolution/convolution_utilities_mpi.h" 
#include <mpi.h>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

#define N_ITERS 20000
#define EPS 1e-8
const int YMIN=0;     
const int XMIN=0;     
const int YMAX=80;      
const int XMAX=80;

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

int main(int argc, char **argv)
{

  int n_ranks, my_rank;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

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
  static Convolution<double> convolve;
  vector<Domain1D<double>> domain_g; // Domain global
  domain_g.push_back(Domain1D<double>(YMIN, YMAX, YMAX-YMIN));
  domain_g.push_back(Domain1D<double>(XMIN, XMAX, XMAX-XMIN));

  if ( domain_g[Y].n_points*domain_g[X].n_points < my_rank )
  {
    if(my_rank == 0)
      printf("There are more processes than discrete points. Terminate program\n");
    MPI_Finalize();
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

    // printf("%d\n", rem);

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
  double dx = 1.0/(XMAX-1);
  double dy = 1.0/(YMAX-1);
  double dx2 = dx*dx;
  double dy2 = dy*dy;
  double dx2i = 1.0 / (6*dx2);
  double dy2i = 1.0 / (6*dy2);
  double dt = min(dx2, dy2)/4.0;
  double k=1.0;
  double factor = k*dt*dx2i;


  // ============================================================================ 
  // Define finite differences kernel
  // ============================================================================ 
  LaplaceOperator9<double> g;

  // ============================================================================ 
  // Initialize subdomain
  // ============================================================================ 
  TestFunction<double>  u(domain_l[Y].n_points, domain_l[X].n_points, InitType::ZERO);
  TestFunction<double> du(domain_l[Y].n_points, domain_l[X].n_points, InitType::ZERO);

  // Apply boundary condition
  {
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
  }

  // Find neighbors and allocate ghost cells
  find_neighbours(my_rank, u.nb, mpi_grid);
  u.allocate_buffers_for_ghost_cells(g);

  int n_recv_req, n_send_req;
  MPI_Status  recv_status[n_neighbors];
  MPI_Status  send_status[n_neighbors];
  MPI_Request send_requests[n_neighbors];
  MPI_Request recv_requests[n_neighbors];

  size_t yy = (u.nb[NO]==-1)?1 :0;
  size_t xx = (u.nb[WE]==-1)?1 :0;
  size_t mm = (u.nb[SO]==-1)?(u.m-1): u.m;
  size_t nn = (u.nb[EA]==-1)?(u.n-1): u.n;

  // ============================================================================ 
  //  Solve
  // ============================================================================ 

  if (my_rank == 0)
  {
    printf("Unsteady 2D Heat equation solved by FTCD \n");
    printf("dx = %5.3e, dy = %5.3e, dt = %5.3e, eps = %5.3e, factor = %5.3e,\n", dx, dy, dt, EPS, factor);
    printf("============================================================================\n");
  }

  //
  // FTCD
  // 

  double exec_time = 0; 
  double comm_time = 0;
  double criterion_time = 0;
  int stride=1000;
  bool done = false;

  double du_max;
  int iter;

  exec_time -= MPI_Wtime();
  for (iter=1; iter<N_ITERS; iter++)
  {
    du_max = 0.0;

    comm_time -= MPI_Wtime();
    u.load_ghost_cells_to_send(g);
    exchange_cells(u, recv_requests, send_requests, n_recv_req, n_send_req, my_rank);
    MPI_Waitall(n_recv_req, &recv_requests[0], &recv_status[0]);
    comm_time += MPI_Wtime();    

    // Convolve
    {
      convolve.B22(g, u, du);
      bool use_buffers = 1;
      convolve.B11(g, u, du, use_buffers);
      convolve.B12(g, u, du, use_buffers);
      convolve.B13(g, u, du, use_buffers);

      convolve.B21(g, u, du, use_buffers);
      convolve.B23(g, u, du, use_buffers);

      convolve.B31(g, u, du, use_buffers);
      convolve.B32(g, u, du, use_buffers);
      convolve.B33(g, u, du, use_buffers);
    }

    // Scale h. h has 0's at the boundaries
    // Compute max h
    // Update u
    for (size_t i=yy; i<mm; i++)
    {
      for (size_t j=xx; j<nn; j++)
      {
        size_t idx =i*u.l+j;
        du.data[idx] = du.data[idx]*factor;

        if(du_max < du.data[idx])
          du_max = du.data[idx];

        u.data[idx] = u.data[idx] + du.data[idx];
      }
    }

    // Check convergence every 'stride' times
    if ((iter % stride) == 0)
    {
      if(my_rank==0)
        printf("%d\n",iter);

      double val = du_max;
      criterion_time -= MPI_Wtime();
      MPI_Allreduce(&val, &du_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      criterion_time += MPI_Wtime();

      if (du_max < EPS ) done = true;
    }

    if(done) break; 


  }

  exec_time += MPI_Wtime();

  //print_by_rank(n_ranks, my_rank, u);

  // print_matrix(u.m, u.n, u.l, u.data);
  
  if (my_rank==0)
  {
    printf("Iterations: %d\n", iter);
    printf("Total execution: %8.3f [ms]\n", exec_time*1000 );
    printf("Total  MPI comm: %8.3f [ms]\n", comm_time*1000);
    printf("MPI_Allreduce  : %8.3f [ms]\n", criterion_time*1000);
    printf("Check conv.    : %3d iter\n", stride);
  }

  MPI_Finalize();
  return 0;
}

