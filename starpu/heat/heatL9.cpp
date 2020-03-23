#include <stdio.h>
#include <vector>
#include <cmath>
#include <chrono>

#define YMAX 20      //
#define XMAX 20      //
#define ITE1 20000   // Max number of iterations
#define EPS  1e-8    // 

using namespace std;

void print_matrix(vector<double> &A, int y, int x)
{
  for (int i=0; i<y; i++)
  {
    for (int j=0; j<x; j++)
      printf("%5.8f ", A[i*x+j]);
    printf("\n");
  }
}

int main(int argc, char **argv)
{
  int ymax = YMAX;
  int xmax = XMAX;
  int itermax = ITE1;
  double eps = EPS;
  double k = 1.0;

  printf("\nUnsteady 2D Heat equation solved by FTCD \n");

  if(argc == 5)
  {
    ymax = atoi(argv[1]);
    xmax = atoi(argv[2]);
    itermax = atoi(argv[3]);
    eps = atof(argv[4]);
  }
  else
  {
    printf("Use default values\n");
  }


  // Initialize
  double dx = 1.0/(xmax-1);
  double dy = 1.0/(ymax-1);
  double dx2 = dx*dx;
  double dy2 = dy*dy;
  double dx2i = 1.0 / (6*dx2);
  double dy2i = 1.0 / (6*dy2);
  double dt = min(dx2, dy2)/4.0;
  double factor = k*dt*dx2i;

  vector<double>  u(ymax*xmax, 0.0); // current iter
  vector<double> un(ymax*xmax, 0.0); // next

  printf("factor: %5.3e, dx = %5.3e, dy = %5.3e, dt = %5.3e, eps = %5.3e, %d, %d, %ld\n", factor, dx, dy, dt, eps, ymax, xmax, u.size());
  // Set boundary conditions
  // WE(NO->SO)=0.0:0.0
  // EA(NO->SO)=1.0:1.0
  // NO(WE->EA)=0.0:dx:1.0
  // SO(WE->EA)=0.0:dx:1.0

  for(int j=1; j<xmax; j++)
  {
     u[j] = u[j-1] + dx; //no
     u[(ymax-1)*xmax+(j)] = u[j];  //so
  }

  for(int i=1; i<ymax; i++)
  {
    int idx = i*xmax-1;
    u[idx] = 1.0;
  }

  chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  // FTCD
  int iter;
  for (iter=1; iter<itermax; iter++)
  {
    double dumax = 0.0;
    for (int i=1; i<ymax-1; i++)
    {
      for (int j=1; j<xmax-1; j++)
      {
        // Compute Laplacian
        int ij = (i)*xmax+(j);
        int we = (i)*xmax+(j-1);
        int ea = (i)*xmax+(j+1);
        int no = (i-1)*xmax+(j);
        int so = (i+1)*xmax+(j);
        int nw = (i-1)*xmax+(j-1);
        int ne = (i-1)*xmax+(j+1);
        int sw = (i+1)*xmax+(j-1);
        int se = (i+1)*xmax+(j+1);

        double Lu = (
           1.0*u[nw] + 4.0*u[no] + 1.0*u[ne]+
           4.0*u[we] -20.0*u[ij] + 4.0*u[ea]+
           1.0*u[sw] + 4.0*u[so] + 1.0*u[se]
            );
        double du = Lu;

        dumax = max(dumax, du);

        un[ij] = du;
      }
    }

    dumax *= factor;
    printf("[0] %f, %u\n", dumax, iter);
    // print_matrix(un, ymax, xmax);
    // printf("\n");

    // Update current iteration
    for (int i=1; i<ymax-1; i++)
    {
      for (int j=1; j<xmax-1; j++)
      {
        int ij = (i)*xmax+(j);
        u[ij] = u[ij] + factor*un[ij];
      }
    }

    // printf("\n");
    // printf("\n");

    if(dumax < eps)
      break;
  }

  // print_matrix(u, ymax, xmax);

  // FILE *f = fopen("reference", "w");
  // if (f == NULL)
  // {
  //   printf("Error opening file!\n");
  //   exit(1);
  // }
  // for (size_t i=0; i<ymax; i++)
  // {
  //   for (size_t j=0; j<xmax; j++)
  //   {
  //     fprintf(f, "%5.5f ", u[i*xmax+j]);
  //   }
  //   fprintf(f, "\n");
  // }
  // fclose(f);

  chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  size_t elapsed_time = chrono::duration_cast<chrono::microseconds>(end - begin).count();
  printf("Iterations: %d\n", iter);
  // printf("Error: %e\n", dumax);
  printf("Elapsed time = %lu [ms]\n", elapsed_time/1000);
  print_matrix(u, ymax, xmax);
  return 0;
}

