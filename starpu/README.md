# README
Implementation of the solution of the 2D unsteady heat equation using task-based
parallelism with StarPU. 

## Solution of the 2D unsteady heat equation

The general form of the heat equation is 
$$\frac{\partial u}{ \partial t} = k \Delta u$$ 
with $u=u(x_{1}, x_{2}, \dots, x_{n})$, $\Delta = \sum_{i=1}^{n}
\frac{\partial^{2} }{\partial x^{2}_{i}}$
the Laplacian in $n$-dimensions and $k$ the coefficient of thermal conductivity.
This equation a unique solution if Dirichlet boundary conditions are
specified.  The solution can be found using the forward-time central-space
method and  requires time and spatial discretization schemes.

The discretized element $(i,j)$ at time $n$ is denoted by  $u^{n}_{i,j}$,
the temporal discretization  is given by
$$\frac{\partial u_{i,j}^{n}}{\partial t} = \frac{\partial u_{i,j}^{n+1} -
\partial u_{i,j}^{n}}{\Delta t}, \label{eqn:dis_time} \tag{1} $$
and the spatial discretization is given by
$$ \frac{\partial^{2} u}{\partial x^{2}} +
  \frac{\partial^{2} u}{\partial y^{2}} =
  \frac{u_{i+1,j}-2u_{i,j}+u_{i-1,j}}{h_{x}^{2}} +
  \frac{u_{i,j+1}-2u_{i,j}+u_{i,j-1}}{h_{y}^{2}}.
  \label{eqn:dis_space} \tag{2}
$$
Combining equations \eqref{eqn:dis_time} and \eqref{eqn:dis_space} , we obtain the recurrence formula
$$ u_{i,j}^{n+1} = u^{n}_{i,j} + k \Delta t 
  \bigg(
    \frac{u_{i+1,j}-2u_{i,j}+u_{i-1,j}}{h_{x}^{2}} +
  \frac{u_{i,j+1}-2u_{i,j}+u_{i,j-1}}{h_{y}^{2}}\bigg).
  \label{eqn:recurrence_formula} \tag{2}
$$
The explicit scheme \eqref{eqn:dis_time} is stable if $h_{x}=h_{y}=h$ and $\Delta t \leq \frac{h^{2}}{4k}$.

## Code
The main purpose of this code is to get experience and test capabilities of
StarPU for an Eulerian scheme since the GENE code uses this approach for the solution of
the gyrokinetic Vlasov equations. 

For this implementation, the 2D Laplacian is treated as a convolution operation
The convolution operation is split into tasks which can be executed by the CPUs
or GPUs available and are scheduled using StarPU.
The computational domain of the heat equation is block-partitioned according to
the number of MPI ranks available.
Each MPI rank exchanges ghost cells with its neighbors to make each subdomain
data independent. 
Cell exchange is also considered as a task and it is also managed with StarPU.

The implementation is done in C++ using templates and STL library.
Implementation of the convolution code with StarPU is also available in Fortran
(CPU only).
The code has been tested with StarPU 1.2.9.
Serial implementations to test correctness of the (periodic/circular)
convolution and the solution of heat equation are also included.



