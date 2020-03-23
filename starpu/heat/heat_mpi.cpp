#include <stdio.h>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <assert.h>
#include <mpi.h>

#define DIMS 2
#define INIT 0 //init
#define END  1 //end
#define Y 0 // y-dim
#define X 1 // x-dim
#define ITER1 20000
#define EPS 1e-8
#define YMIN 0      
#define XMIN 0      
#define YMAX 80      
#define XMAX 80      
#define NO 0 // 
#define WE 1 // 
#define EA 2 // 
#define SO 3 // 


#define PRINT_DEBUG 0

using namespace std;

struct Domain2D
{
  vector<vector<int>> dim;  // Domain limits in each dimension [DIM][init, end)
  vector<int> num_points;   // Num. of  discrete points without ghost cells

  vector<int> np;          // Index of neighbour process
  vector<int> gcmem_r;     // Memory index of the received cells
  vector<int> gcmem_s;     // Memory index of the sent cells
  int memstride;           // Memory stride (row-major)

  MPI_Datatype mpivector[DIMS];  // MPI data type used to exchange cells in each dimention

  Domain2D()
  {
    dim = vector<vector<int>>(DIMS, vector<int>(2, 0));
    num_points = vector<int>(DIMS, 0);
    np = vector<int>(4,-1);
  }

  void initialize(int yinit, int yend, int xinit, int xend)
  {
    dim[Y][INIT] = yinit; dim[Y][END] = yend;
    dim[X][INIT] = xinit; dim[X][END] = xend;
    num_points[Y] = dim[Y][END] - dim[Y][INIT];
    num_points[X] = dim[X][END] - dim[X][INIT];
  }

  int mempos(int i, int j)
  {
    return i*memstride+j; 
  }

  void init_gcmem()
  {
    gcmem_r.resize(4,-1);
    gcmem_s.resize(4,-1);
    int yo, xo;
    yo = (np[SO]==-1)?0:1; 
    xo = (np[WE]==-1)?0:1;
    if(np[SO]!=-1) gcmem_r[SO]=mempos(0, 1);
    if(np[WE]!=-1) gcmem_r[WE]=mempos(1, 0);
    if(np[NO]!=-1) gcmem_r[NO]=mempos(num_points[Y]+yo, 1);
    if(np[EA]!=-1) gcmem_r[EA]=mempos(1, num_points[X]+xo);

    yo = (np[SO]==-1)?1:0; 
    xo = (np[WE]==-1)?1:0;
    if(np[SO]!=-1) gcmem_s[SO]=mempos(1, 1);
    if(np[WE]!=-1) gcmem_s[WE]=mempos(1, 1);
    if(np[NO]!=-1) gcmem_s[NO]=mempos(num_points[Y]-yo, 1);
    if(np[EA]!=-1) gcmem_s[EA]=mempos(1, num_points[X]-xo);
  }

  void initialize_mpi_data_types()
  {
    // horizontal vector, cell exchange along Y-dir
    int blocks, length, stride;
    blocks = 1;
    length = num_points[X];
    if(np[EA]==-1) length--;
    if(np[WE]==-1) length--;
    stride = memstride;
    MPI_Type_vector(blocks, length, stride, MPI_DOUBLE, &mpivector[Y]);
    MPI_Type_commit(&mpivector[Y]);

    // vertical vector, cell exchange along X-dir
    blocks = num_points[Y];
    if(np[NO]==-1) blocks--;
    if(np[SO]==-1) blocks--;
    length = 1;
    stride = memstride;
    MPI_Type_vector(blocks, length, stride, MPI_DOUBLE, &mpivector[X]);
    MPI_Type_commit(&mpivector[X]);
  }

  string to_string()
  {
    return string("y[" + std::to_string(dim[Y][INIT]) + "," + std::to_string(dim[Y][END])+")"+ " x[" + std::to_string(dim[Y][INIT]) + "," + std::to_string(dim[Y][END])+")");
  }
};

#if PRINT_DEBUG
void print_matrix(vector<double> *A, size_t *ny, size_t *nx, string *str)
{
  if(A)
  {
    for (int i=0; i<*ny; i++)
    {
      for (int j=0; j<*nx; j++)
      {
        printf("%1.2f ", (*A)[i*(*nx)+j]);
      }
      printf(" <- %d \n", i);
    }
  }

  if(str)
  {
    printf("%s", str->c_str());
  }
}

void print_by_rank(MPI_Comm &MPI_COMM_0, int world, int rank, vector<double> *A, size_t *ny, size_t *nx, string *str)
{
  MPI_Barrier(MPI_COMM_0);
  int message=0;
  if (rank == 0)
  {
    print_matrix(A, ny, nx, str);
    MPI_Send(&message, 1, MPI_INT, 1, 0, MPI_COMM_0);
  } else {
    int buffer;
    MPI_Status status;
    MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_0, &status);
    MPI_Get_count(&status, MPI_INT, &buffer);
    if (buffer == 1)
    {
      print_matrix(A, ny, nx, str);
      MPI_Recv(&message, buffer, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_0, &status);
      if (rank + 1 != world)
      {
        MPI_Send(&message, 1, MPI_INT, ++rank, 0, MPI_COMM_0);
      }
    }
  }
}
#endif

void domain_decomposition(Domain2D &domain, MPI_Comm &MPI_COMM_0, int &world, int &rank, int pgrid[DIMS], int pcoord[DIMS], Domain2D &sub)
{
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //
  // Distribute nodes in cartesian grid
  //
  if( domain.num_points[Y]*domain.num_points[X] < rank )
  {
    if(rank == 0)
      printf("There are more processes than discrete points. Terminate program\n");
    MPI_Finalize();
    exit(0);
  }

  // Let MPI create a cartesian grid
  for(int i=0; i<DIMS; i++) pgrid[i]=0;
  MPI_Dims_create(world, DIMS, pgrid);

  if (rank  == 0)
  {
    printf("Domain: %s \n", domain.to_string().c_str());
    printf("Use %d proc. in a cart. grid y=%d, x=%d\n", world, pgrid[Y], pgrid[X]);
  }

  // Create a new MPI context
  int period[DIMS]; 
  for(int i=0; i<DIMS; i++) period[i]=0;
  MPI_Cart_create(MPI_COMM_WORLD, DIMS, pgrid, period, 1, &MPI_COMM_0);

  // Get rank and coord in MPI_COMM_0
  MPI_Comm_rank(MPI_COMM_0, &rank);
  for(int i=0; i<DIMS; i++) pcoord[i]=0;
  MPI_Cart_coords(MPI_COMM_0, rank, DIMS, pcoord);

  //
  // Determine neighbours
  //
  MPI_Cart_shift(MPI_COMM_0, X, 1, &sub.np[WE], &sub.np[EA]);
  MPI_Cart_shift(MPI_COMM_0, Y, 1, &sub.np[SO], &sub.np[NO]);

  //
  // Determine sub
  // 
  vector<vector<int>> dim(DIMS,vector<int>(2, 0));
  for (int i = 0; i<DIMS; i++)
  {
    int len = (domain.num_points[i]) / pgrid[i];
    int rem = (domain.num_points[i]) - (pgrid[i] * len);

    if(pcoord[i] < rem)
    {
      dim[i][INIT] = domain.dim[i][INIT] +pcoord[i]*len +pcoord[i];
      dim[i][END]= dim[i][INIT] + len+1;
    }
    else
    {
      dim[i][INIT] = domain.dim[i][INIT] + (pcoord[i])*len + rem;
      dim[i][END]= dim[i][INIT] + len;
    }
  }

  sub.initialize(dim[Y][INIT], dim[Y][END], dim[X][INIT], dim[X][END]);

  // Add ghost cells
  sub.memstride = sub.num_points[X];
  if(sub.np[WE]>-1) sub.memstride++;
  if(sub.np[EA]>-1) sub.memstride++;

  sub.init_gcmem();
  sub.initialize_mpi_data_types();
}

void exchange_cells( MPI_Comm &MPI_COMM_0, MPI_Status status[4], MPI_Request req[4], Domain2D &s, vector<double> &u)
{
  // Y-dir
  int i=0;
  if(s.np[SO]!=-1) MPI_Irecv(&u[s.gcmem_r[SO]], 1, s.mpivector[Y], s.np[SO], MPI_ANY_TAG, MPI_COMM_0, &req[i++]);
  if(s.np[NO]!=-1) MPI_Irecv(&u[s.gcmem_r[NO]], 1, s.mpivector[Y], s.np[NO], MPI_ANY_TAG, MPI_COMM_0, &req[i++]);
  if(s.np[SO]!=-1) MPI_Isend(&u[s.gcmem_s[SO]], 1, s.mpivector[Y], s.np[SO], 0, MPI_COMM_0, &req[i++]);
  if(s.np[NO]!=-1) MPI_Isend(&u[s.gcmem_s[NO]], 1, s.mpivector[Y], s.np[NO], 0, MPI_COMM_0, &req[i++]);
  MPI_Waitall(i, req, status);

  // X-dir
  int j=0;
  if(s.np[WE]!=-1) MPI_Irecv(&u[s.gcmem_r[WE]], 1, s.mpivector[X], s.np[WE], MPI_ANY_TAG, MPI_COMM_0, &req[j++]);
  if(s.np[EA]!=-1) MPI_Irecv(&u[s.gcmem_r[EA]], 1, s.mpivector[X], s.np[EA], MPI_ANY_TAG, MPI_COMM_0, &req[j++]);
  if(s.np[WE]!=-1) MPI_Isend(&u[s.gcmem_s[WE]], 1, s.mpivector[X], s.np[WE], 0, MPI_COMM_0, &req[j++]);
  if(s.np[EA]!=-1) MPI_Isend(&u[s.gcmem_s[EA]], 1, s.mpivector[X], s.np[EA], 0, MPI_COMM_0, &req[j++]);
  MPI_Waitall(j, req, status);
}

void solve(int world, int rank, Domain2D &d, Domain2D &s, MPI_Comm &MPI_COMM_0)
{
  double dx   = 1.0f / (d.num_points[X]);
  double dy   = 1.0f / (d.num_points[Y]);
  double dx2  = dx * dx;     
  double dy2  = dy * dy;
  double dx2i = 1.0f / dx2; 
  double dy2i = 1.0f / dy2;
  double dt = min(dx2, dy2) / 4.0;
  double eps = EPS;
  double k=1.0;

  if (rank == 0)
  {
    printf("Unsteady 2D Heat equation solved by FTCD \n");
    printf("dx = %5.3e, dy = %5.3e, dt = %5.3e, eps = %5.3e\n", dx, dy, dt, eps);
    printf("============================================================================\n");
  }

  //
  // Initialize domain
  // 
  size_t nx = (s.num_points[Y] + ((s.np[NO]>-1)?1:0) + ((s.np[SO]>-1)?1:0));
  size_t ny = (s.num_points[X] + ((s.np[WE]>-1)?1:0) + ((s.np[EA]>-1)?1:0));
  size_t nn = nx*ny;

  vector<double>  u(nn, 0.0);

  if(s.dim[Y][INIT] == d.dim[Y][INIT])
  {
    int yo = (s.np[SO]==-1)?0:1; // mem. offset
    int xo = (s.np[WE]==-1)?0:1;
    int i=0;
    for(int j=0; j<s.num_points[X]; j++)
    {
      u[s.mempos(i+yo,j+xo)] = (s.dim[X][INIT]+j)*dx; //south
    }
  }

  if(s.dim[Y][END] == d.dim[Y][END])
  {
    int yo = (s.np[SO]==-1)?0:1; // mem. offset
    int xo = (s.np[WE]==-1)?0:1;
    int i = s.num_points[Y]-1; 
    for(int j=0; j<s.num_points[X]; j++)
    {
      u[s.mempos(yo+i,j+xo)] = (s.dim[X][INIT]+j)*dx; //north
    }
  }

  if(s.dim[X][END] == d.dim[X][END])
  {
    int yo = (s.np[SO]==-1)?0:1; // mem. offset
    int xo = (s.np[WE]==-1)?0:1;
    int j = s.num_points[X]-1;
    for(int i=0; i<s.num_points[Y]; i++)
      u[s.mempos(yo+i,j+xo)] = 1.0; // east
  }

  vector<double> un(u);

  //
  // FTCD
  // 
  MPI_Status status[4];
  MPI_Request request[4];

  double exec_time = 0; 
  double comm_time  = 0;
  double criterion_time = 0;

  int stride=1;
  int iter;
  int itermax=ITER1;
  bool done = false;

  exec_time -= MPI_Wtime();
  double dumax;

  int lengthY = s.num_points[Y]+1;
  if(s.np[NO]==-1) lengthY--;
  if(s.np[SO]==-1) lengthY--;

  int lengthX = s.num_points[X]+1;
  if(s.np[EA]==-1) lengthX--;
  if(s.np[WE]==-1) lengthX--;

  for (iter=1; iter<itermax; iter++)
  {
    dumax = 0.0;
    for (int i=1; i<lengthY; i++)
    {
      for (int j=1; j<lengthX; j++)
      {
        // Compute Laplacian
        int ij = s.mempos(i,j);
        int we = s.mempos(i,j-1);
        int ea = s.mempos(i,j+1);
        int no = s.mempos(i+1,j);
        int so = s.mempos(i-1,j);
        double Lu = (u[we]+u[ea]-2.0*u[ij])*dx2i + (u[no]+u[so]-2.0*u[ij])*dy2i;
        double du = k*dt*Lu;

        dumax = max(dumax, du);

        un[ij] = u[ij] + du;
      }
    }

    // Update cu
    u = un;

    // Check convergence every 'stride' times
    if ((iter % stride) == 0)
    {
      double val = dumax;
      criterion_time -= MPI_Wtime();
      MPI_Allreduce(&val, &dumax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_0);
      criterion_time += MPI_Wtime();

      if (dumax < eps) done = true;
    }

    if(done) break; 

    comm_time -= MPI_Wtime();
    exchange_cells(MPI_COMM_0, status, request, s, u);
    comm_time += MPI_Wtime();    
  }    

  exec_time += MPI_Wtime();

  if (rank==0)
  {
    printf("Iterations: %d\n", iter);
    printf("Total execution: %8.3f [ms]\n", exec_time*1000 );
    printf("Total  MPI comm: %8.3f [ms]\n", comm_time*1000);
    printf("MPI_Allreduce  : %8.3f [ms]\n", criterion_time*1000);
    printf("Check conv.    : %3d iter\n", stride);
  }
}

#if PRINT_DEBUG
void test_cell_exchange(MPI_Comm &MPI_COMM_0, Domain2D &s, int world, int rank)
{
  size_t nx = (s.num_points[Y] + ((s.np[NO]>-1)?1:0) + ((s.np[SO]>-1)?1:0));
  size_t ny = (s.num_points[X] + ((s.np[WE]>-1)?1:0) + ((s.np[EA]>-1)?1:0));
  size_t nn = nx*ny;

  vector<double>  u(nn, 0.0);

  int yo = (s.np[SO]==-1)?0:1; // mem. offset
  int xo = (s.np[WE]==-1)?0:1;
  int i = 0;
  for (int j=0; j<s.num_points[X]; j++)
    u[s.mempos(i+yo,j+xo)] = rank+1;

  i = s.num_points[Y]-1; 
  for (int j=0; j<s.num_points[X]; j++)
    u[s.mempos(i+yo,j+xo)] = rank+1;


  int j = s.num_points[X]-1;
  for(int i=0; i<s.num_points[Y]; i++)
    u[s.mempos(yo+i,j+xo)] = rank+1;

  j = 0;
  for(int i=0; i<s.num_points[Y]; i++)
    u[s.mempos(yo+i,j+xo)] = rank+1;

  s.init_gcmem();
  s.initialize_mpi_data_types();

  MPI_Status status[4];
  MPI_Request request[4];
  exchange_cells(MPI_COMM_0, status, request, s, u);
  print_by_rank(MPI_COMM_0, world, rank, &u, &ny, &nx, NULL);
}
#endif

int main(int argc, char **argv)
{
  Domain2D domain, subdomain;
  domain.initialize(YMIN,YMAX, XMIN, XMAX);

  MPI_Init(&argc, &argv);

  MPI_Comm MPI_COMM_0;
  int world, rank, pgrid[DIMS], pcoord[DIMS];

  // Domain decomposition
  domain_decomposition(domain, MPI_COMM_0, world, rank, pgrid, pcoord, subdomain);

#if PRINT_DEBUG
  string str = "Node " + to_string(rank) + "(" + to_string(pcoord[Y]) + ", "+to_string(pcoord[X]) +"), subdomain: "+ subdomain.to_string() +"\n";
  print_by_rank(MPI_COMM_0, world, rank, NULL, NULL, NULL, &str);
#endif

  // Solve
  solve(world, rank, domain, subdomain, MPI_COMM_0); 

  MPI_Barrier(MPI_COMM_0);
  MPI_Finalize();

  return 0;
}
