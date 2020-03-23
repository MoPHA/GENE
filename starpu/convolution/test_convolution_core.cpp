#include "convolution_core.h"

#define PRINT_DEBUG  0
#define MPI_ENABLED 1

#if MPI_ENABLED 
#include "convolution_utilities_mpi.h"
#endif

void test_convolution(size_t r, size_t m, size_t n)
{
  InitType f_inittype = InitType::RAND;
  InitType h_inittype = InitType::ZERO;

  if (0)
  {
    // ---------------------------------------------------------------------------- 
    // Compute convolution (serial) 
    // ---------------------------------------------------------------------------- 
    const TestKernel<double> g(r);        // kernel

    TestFunctionConv<double> ff(m, n, r, f_inittype);   // input (reference)
    TestFunction<double>     hh(m, n,    h_inittype);  // output (reference)
    compute_serial_convolution(ff, g, hh);
    //return;

    // ---------------------------------------------------------------------------- 
    // Compute convolution (task based) 
    // ---------------------------------------------------------------------------- 
    TestFunction<double> f(m, n, f_inittype);
    TestFunction<double> h(m, n, h_inittype); // MUST be initialized to 0 !!

    f.nb = vector<int>(n_neighbors, 0); f.nb[4]=-1;
    f.allocate_buffers_for_ghost_cells(g);
    f.load_ghost_cells_to_send(g);

    // Exchange ghost cells
    my_swap(f.gr[NW], f.gs[SE]);
    my_swap(f.gr[NO], f.gs[SO]);
    my_swap(f.gr[NE], f.gs[SW]);
    my_swap(f.gr[WE], f.gs[EA]);
    my_swap(f.gr[EA], f.gs[WE]);
    my_swap(f.gr[SW], f.gs[NE]);
    my_swap(f.gr[SO], f.gs[NO]);
    my_swap(f.gr[SE], f.gs[NW]);

    bool use_buffers = 1;
    Convolution<double> convolve;
    convolve.B11(g, f, h, use_buffers);
    convolve.B12(g, f, h, use_buffers);
    convolve.B13(g, f, h, use_buffers);
    convolve.B21(g, f, h, use_buffers);
    convolve.B22(g, f, h);
    convolve.B23(g, f, h, use_buffers);
    convolve.B31(g, f, h, use_buffers);
    convolve.B32(g, f, h, use_buffers);
    convolve.B33(g, f, h, use_buffers);

    // print_matrix(f.m, f.n, f.l, f.data);
    // print_matrix(h.m, h.n, h.l, h.data);
    double error = compute_error(hh.data, h.data, h.m, h.n);
    printf("Single block - Abs error: %5.3f\n\n", error);
  }

  if(0)
  {
    //---------------------------------------------------------------------------- 
    // Convolution into tasks
    //---------------------------------------------------------------------------- 

    // Size of submatrices are (ms, ns)
    size_t ms=300;   
    size_t ns=300;    

    // Number of submatrices in which rank-matrix is dicomposed 
    size_t ny = 1; 
    size_t nx = 1;  

    // Size of rank-matrices are (m, m)
    size_t m = ny * ms; // size of the matrix in one mpi rank (rank matrix)
    size_t n = nx * ns;

    size_t r=3;

    assert(ns-1>r);
    assert(ms-1>r);

    const TestKernel<double> g(r);  // kernel

    // Rank-matrices 
    TestFunctionConv<double> ff(m, n, r, f_inittype); // input (reference)
    TestFunction<double>     hh(m, n,    h_inittype); // output(reference)

    // Perform convolution  
    compute_serial_convolution(ff, g, hh);

    TestFunction<double> f(m, n, f_inittype);   
    TestFunction<double> h(m, n, h_inittype);  

    // For this testing, make f = ff
    my_swap(ff.data, f.data);
    my_swap(ff.m, f.m);
    my_swap(ff.n, f.n);
    my_swap(ff.l, f.l);

    printf("Convolution: rank matrix (%ld, %ld), g(%ld, %ld), tasks(%ld, %ld)=%ld, task_size(%ld,%ld)\n",
        m, n, g.m, g.n, ny, nx, ny*nx, ms, ns);

    size_t g_mem = g.get_mem_size();
    size_t f_mem = f.get_mem_size();
    size_t h_mem = h.get_mem_size();

    size_t hh_mem = hh.get_mem_size();
    size_t ff_mem = ff.get_mem_size();

    // Sub matrices
    vector< TestFunction<double>* > f_sub(ny * nx , NULL);
    vector< TestFunction<double>* > h_sub(ny * nx , NULL);

    size_t f_sub_mem, h_sub_mem;
    rank_matrix_to_submatrices(g, f, f_sub, h_sub, f_sub_mem, h_sub_mem, ms, ns, ny, nx);

    char buf[32];
    pretty_bytes(buf, g_mem); printf("g:%s\n", buf);
    pretty_bytes(buf, f_mem); printf("f: %s\n", buf);
    pretty_bytes(buf, h_mem); printf("h: %s\n", buf);
    pretty_bytes(buf, hh_mem); printf("hh: %s\n", buf);
    pretty_bytes(buf, f_sub_mem); printf("f_sub_mem:%s\n", buf);
    pretty_bytes(buf, h_sub_mem); printf("h_sub_mem:%s\n", buf);

    // for (size_t i=0; i<f_sub.size(); i++)
    //   print_matrix(f_sub[i]->m, f_sub[i]->n, f_sub[i]->l, f_sub[i]->data);

    for (size_t i=0; i<f_sub.size(); i++)
      compute_convolution_sub(
          g.data, g.m, g.n, g.l,
          f_sub[i]->data, f_sub[i]->m, f_sub[i]->n, f_sub[i]->l,
          h_sub[i]->data, h_sub[i]->m, h_sub[i]->n, h_sub[i]->l);

    submatrices_into_rank_matrix(h, h_sub, ms, ns, ny, nx);

    // for (size_t i=0; i<h_sub.size(); i++)
    //   print_matrix(h_sub[i]->m, h_sub[i]->n, h_sub[i]->l, h_sub[i]->data);
    // print_matrix(g.m, g.n, g.l, g.data);
    // print_matrix(f.m, f.n, f.l, f.data);
    // print_matrix(h.m, h.n, h.l, h.data);
    // print_matrix(hh.m, hh.n, hh.l, hh.data);

    double error = compute_error(hh.data, h.data, h.m, h.n);
    printf("Convolution into tasks: Abs error: %5.3f\n\n", error);

    for (size_t i=0; i<f_sub.size(); i++)
    {
      if(f_sub[i]){ delete f_sub[i]; f_sub[i]=NULL;}
      if(h_sub[i]){ delete h_sub[i]; h_sub[i]=NULL;}
    }
  }

  if (1)
  {
    // ---------------------------------------------------------------------------- 
    // Simulate mpi 
    // ---------------------------------------------------------------------------- 
    const int n_ranks = 9;
    const TestKernel<double> g(r);

    // Distribute nodes in a cartesian grid and fnd neighbours for each node
    int mpi_grid[DIMS];
    int n_ranks_y = sqrt(n_ranks); //(n_ranks % 2 )? 1: 2;
    int n_ranks_x = sqrt(n_ranks); //(n_ranks / n_ranks_y);
    mpi_grid[Y] = n_ranks_y;
    mpi_grid[X] = n_ranks_x;

    // Create discretization grid
    size_t msup = n_ranks_y * m;  // number of points in y-dir
    size_t nsup = n_ranks_x * n;  // number of points in x-dir

    printf("Simulate MPI: Proc(%d, %d) Globalsize(%ld, %ld), Ranksize(%ld, %ld), r=%ld \n", 
        n_ranks_y, n_ranks_x, msup, nsup, m, n, r);

    // ---------------------------------------------------------------------------- 
    // Compute convolution (serial) 
    // ---------------------------------------------------------------------------- 
    TestFunctionConv<double> ff0(msup, nsup, r, f_inittype);
    TestFunction<double>     hh0(msup, nsup,    h_inittype);

    compute_serial_convolution(ff0, g, hh0);

    // ---------------------------------------------------------------------------- 
    // Compute blocked convolution
    // ---------------------------------------------------------------------------- 
    
    Convolution<double> convolve;
    TestFunction<double> fsup(msup, nsup, f_inittype);
    TestFunction<double> hsup(msup, nsup, h_inittype);

    // Create functions for each block
    vector<TestFunction<double>*> f;
    vector<TestFunction<double>*> h;
    for (int i=0; i<n_ranks; i++)
    {
      f.push_back(new TestFunction<double>(m, n, InitType::NONE));
      h.push_back(new TestFunction<double>(m, n, h_inittype));
    }

    //Distribute f among processors
    for (int i=0; i<n_ranks_y; i++)
    {
      for (int j=0; j<n_ranks_x; j++)
      {
        int irank = i*n_ranks_x+j;
        size_t offset_i = i*m;
        size_t offset_j = j*n;

        for (size_t ii=0; ii<m; ii++)
        {
          for (size_t jj=0; jj<n; jj++)
          {
            size_t iwrite = ii*f[irank]->l +jj;
            size_t iread = (offset_i+ii)*fsup.l +(offset_j+jj);
            f[irank]->data[iwrite] = fsup.data[iread];
          }
        }
      }
    }

    for (int irank=0; irank<n_ranks; irank++)
    {
      find_neighbours_periodic(irank, f[irank]->nb, mpi_grid);

      f[irank]->allocate_buffers_for_ghost_cells(g);

      f[irank]->load_ghost_cells_to_send(g);

      // printf(" NW:(%2d), NO:(%2d) NE:(%2d) \n WE:(%2d), ME:(%2d) EA:(%2d) \n SW:(%2d), SO:(%2d) SE:(%2d) \n\n",
      //     f[irank]->nb[NW], f[irank]->nb[NO], f[irank]->nb[NE], f[irank]->nb[WE], irank,
      //     f[irank]->nb[EA], f[irank]->nb[SW], f[irank]->nb[SO], f[irank]->nb[SE]);
    }

    for (int irank=0; irank<n_ranks; irank++)
    {
      my_swap(f[irank]->gr[NW], f[ f[irank]->nb[NW] ]->gs[SE]);
      my_swap(f[irank]->gr[NO], f[ f[irank]->nb[NO] ]->gs[SO]);
      my_swap(f[irank]->gr[NE], f[ f[irank]->nb[NE] ]->gs[SW]);
      my_swap(f[irank]->gr[WE], f[ f[irank]->nb[WE] ]->gs[EA]);
      my_swap(f[irank]->gr[EA], f[ f[irank]->nb[EA] ]->gs[WE]);
      my_swap(f[irank]->gr[SW], f[ f[irank]->nb[SW] ]->gs[NE]);
      my_swap(f[irank]->gr[SO], f[ f[irank]->nb[SO] ]->gs[NO]);
      my_swap(f[irank]->gr[SE], f[ f[irank]->nb[SE] ]->gs[NW]);
    }

    for (int irank=0; irank<n_ranks; irank++)
    {
      bool use_buffers = 1;
      convolve.B11(g, *f[irank], *h[irank], use_buffers);
      convolve.B12(g, *f[irank], *h[irank], use_buffers);
      convolve.B13(g, *f[irank], *h[irank], use_buffers);
      convolve.B21(g, *f[irank], *h[irank], use_buffers);
      convolve.B22(g, *f[irank], *h[irank]);
      convolve.B23(g, *f[irank], *h[irank], use_buffers);
      convolve.B31(g, *f[irank], *h[irank], use_buffers);
      convolve.B32(g, *f[irank], *h[irank], use_buffers);
      convolve.B33(g, *f[irank], *h[irank], use_buffers);
    }

    // Join outputs
    for (int i=0; i<n_ranks_y; i++)
    {
      for (int j=0; j<n_ranks_x; j++)
      {
        int irank = i*n_ranks_x+j;
        size_t offset_i = i*m;
        size_t offset_j = j*n;
        for (size_t ii=0; ii<m; ii++)
        {
          for (size_t jj=0; jj<n; jj++)
          {
            size_t iwrite = (offset_i+ii)*hsup.l +(offset_j+jj);
            size_t iread = ii*h[irank]->l+jj;
            hsup.data[iwrite] = h[irank]->data[iread];
          }
        }
      }
    }

    // print_matrix(hh0.m, hh0.n, hh0.l, hh0.data);
    // print_matrix(hsup.m, hsup.n, hsup.l, hsup.data);

    double error = compute_error(hh0.data, hsup.data, hsup.m, hsup.n);
    printf("Blocked convolution: Abs error: %5.5f\n\n", error);

    for (int i=0; i<n_ranks; i++)
    {
      delete f[i];  f[i]=NULL;
      delete h[i]; h[i]=NULL;
    }
  }
}

#if MPI_ENABLED
void test_convolution_with_mpi_non_blocking(size_t r, size_t m, size_t n)
{
  InitType f_inittype = InitType::RAND;
  InitType h_inittype = InitType::ZERO;

  int n_ranks, my_rank;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

  if(n_ranks < 2)
  {
    fprintf(stderr, "Must use two or more processes for this example\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Distribute nodes in a cartesian grid and find neighbours for each node
  int mpi_grid[DIMS];
  int n_ranks_y = sqrt(n_ranks); //(n_ranks % 2 )? 1: 2;
  int n_ranks_x = sqrt(n_ranks); //(n_ranks / n_ranks_y);
  mpi_grid[Y] = n_ranks_y;
  mpi_grid[X] = n_ranks_x;

  // Create discretization grid
  size_t msup = n_ranks_y*m;  // number of points in y-dir
  size_t nsup = n_ranks_x*n;  // number of points in x-dir

  const TestKernel<double> g(r);

  vector<TestFunction<double>*> f;
  vector<TestFunction<double>*> h;

  vector<MPI_Request> request(max(16, n_ranks));
  vector<MPI_Status> status(request.size());

  int ireq = 0;
  if(my_rank == 0)
  {
    printf("MPI: Proc(%d, %d) Globalsize(%ld, %ld), Ranksize(%ld, %ld), r=%ld \n", n_ranks_y, n_ranks_x, msup, nsup, m, n, r);

    // Create send / receive buffers
    for (size_t i=0; i<n_ranks; i++)
    {
      f.push_back(new TestFunction<double>(m, n, InitType::NONE));
      h.push_back(new TestFunction<double>(m, n, h_inittype));
    }

    TestFunction<double> ff(msup, nsup, f_inittype);
    ff.dump_data_to_file(string("ref"), my_rank);

    // Fill in buffers according its my_rank
    for (int i=0; i<n_ranks_y; i++)
    {
      for (int j=0; j<n_ranks_x; j++)
      {
        int mpi_rank = i*n_ranks_x + j;
        size_t offset_i = i * m;
        size_t offset_j = j * n;

        for (size_t i=0; i<m; i++)
        {
          for (size_t j=0; j<n; j++)
          {
            f[mpi_rank]->data[i*f[mpi_rank]->l+j] = 
              ff.data[(offset_i+i)*ff.l + (offset_j+j)];
          }
        }
        // f[mpi_rank]->dump_data_to_file(string("out"), mpi_rank);
      }
    }

    // Distribute data
    for (int i=0; i<n_ranks_y; i++)
    {
      for (int j=0; j<n_ranks_x; j++)
      {
        int mpi_rank = i*n_ranks_x+j;

        if(mpi_rank!=0) 
        {
          MPI_Send( (void*)f[mpi_rank]->data, f[mpi_rank]->get_num()*sizeof(double), MPI_BYTE, mpi_rank, 0, MPI_COMM_WORLD);
          // MPI_Isend((void*)f[mpi_rank]->data, f[mpi_rank]->get_num()*sizeof(double), MPI_BYTE, mpi_rank, mpi_rank, MPI_COMM_WORLD, &request[ireq++]);
        }
      }
    }

    for (int i=n_ranks; i>=1; i--)
    {
      delete f[i];
      f.pop_back();
    }
  }
  else
  {
    f.push_back(new TestFunction<double>(m, n, InitType::NONE));
    h.push_back(new TestFunction<double>(m, n, h_inittype));

    // Receive data from master
    // MPI_Recv( (void*)f[0]->data, f[0]->get_num()*sizeof(double), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Irecv( (void*)f[0]->data, f[0]->get_num()*sizeof(double), MPI_BYTE, 0, 0, MPI_COMM_WORLD, &request[ireq++]);
  }

  //printf("Rank %ld: %ld\n",my_rank, ireq);
  MPI_Waitall(ireq, &request[0], &status[0]);

#if PRINT_DEBUG
  printf("I got my data %d/%d\n", my_rank, n_ranks-1);
#endif

  //---------------------------------------------------------------------------- 
  // Compute blocked convolution
  // ---------------------------------------------------------------------------- 

  // determine neighbors
  find_neighbours_periodic(my_rank, f[0]->nb, mpi_grid);
  f[0]->allocate_buffers_for_ghost_cells(g);


  int n_recv_req, n_send_req;
  MPI_Status  recv_status[n_neighbors];
  MPI_Status  send_status[n_neighbors];
  MPI_Request send_requests[n_neighbors];
  MPI_Request recv_requests[n_neighbors];
  f[0]->load_ghost_cells_to_send(g);

  exchange_cells(*f[0], recv_requests, send_requests, n_recv_req, n_send_req, my_rank);
  MPI_Waitall(n_recv_req, &recv_requests[0], &recv_status[0]);


#if PRINT_DEBUG
  printf("Ghost cells exchanged! %d\n", my_rank);
#endif

  // print_by_rank(n_ranks, my_rank, *f[0]);

  Convolution<double> convolve;

  bool use_buffers=1;
  convolve.B11(g, *f[0], *h[0], use_buffers);
  convolve.B12(g, *f[0], *h[0], use_buffers);
  convolve.B13(g, *f[0], *h[0], use_buffers);
  convolve.B21(g, *f[0], *h[0], use_buffers);
  convolve.B22(g, *f[0], *h[0]);
  convolve.B23(g, *f[0], *h[0], use_buffers);
  convolve.B31(g, *f[0], *h[0], use_buffers);
  convolve.B32(g, *f[0], *h[0], use_buffers);
  convolve.B33(g, *f[0], *h[0], use_buffers);

  // Send h to master thread
  if(my_rank == 0)
  {
    for (int mpi_rank=1; mpi_rank<n_ranks; mpi_rank++)
    {
      MPI_Recv( (void*)h[mpi_rank]->data, h[mpi_rank]->get_num()*sizeof(double), MPI_BYTE, mpi_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Join outputs
    TestFunction<double> hsup(msup, nsup, h_inittype);
    for (int i=0; i<n_ranks_y; i++)
    {
      for (int j=0; j<n_ranks_x; j++)
      {
        int mpi_rank = i*n_ranks_x+j;
        size_t data_offset_y = i*m;
        size_t data_offset_x = j*n;
        for (size_t ii=0; ii<m; ii++)
        {
          for (size_t jj=0; jj<n; jj++)
          {
            size_t iwrite = (data_offset_y+ii)*nsup +(data_offset_x+jj);
            size_t iread = ii*n+jj;
            hsup.data[iwrite] = h[mpi_rank]->data[iread];
          }
        }
      }
    }

    // ---------------------------------------------------------------------------- 
    // Compute convolution (serial) 
    // ---------------------------------------------------------------------------- 

    TestFunctionConv<double> ff(msup, nsup, r, f_inittype);
    TestFunction<double> hh(msup, nsup, h_inittype);
    compute_serial_convolution(ff, g, hh);

    // print_matrix(hh.m, hh.n, hh.l, hh.data);
    // print_matrix(hsup.m, hsup.n, hsup.l, hsup.data);

    double error = compute_error(hh.data, hsup.data, hsup.m, hsup.n);
    printf("Blocked convolution: Abs error: %5.3f\n\n", error);

  }
  else
  {
    MPI_Send( (void*)h[0]->data, h[0]->get_num()*sizeof(double), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  for (int i=0; i<f.size(); i++)
  {
    delete f[i];
    delete h[i];
  }

  MPI_Finalize();
}
#endif

int main (int argr, char** argv)
{
  size_t r = 1;
  size_t m = 10;
  size_t n = 10;

  assert(n-1>r);
  assert(m-1>r);

#if MPI_ENABLED  
  test_convolution_with_mpi_non_blocking(r, m, n);
#else
  test_convolution(r, m, n);
#endif

}
