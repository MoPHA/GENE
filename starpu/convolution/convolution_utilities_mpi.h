#ifndef _CONVOLUTION_CORE_MPI_H
#define _CONVOLUTION_CORE_MPI_H

#include "convolution_kernels.h"
#include "convolution_core.h"
#include <stdlib.h>
#include <limits>
#include "mpi.h"

using namespace std;

  template<typename T>
void exchange_cells(TestFunction<T> &f,
    MPI_Request recv_requests[n_neighbors],
    MPI_Request send_requests[n_neighbors],
    int &n_recv_req, int &n_send_req,
    int &my_rank)
{
  /*
     MPI_Isend(void *buff, int count, MPI_Datatype type,
     int dest, int tag, int comm, MPI_Request *request );

     Send a point-to-point message ASYNCHRONOUSLY to process dest in the communication group comm
     The message is stored at memory location buff and consists of count items of datatype type
     The message will be tagged with the tag value tag
     The MPI_Isend() function will return IMMEDIATELY
     The current sending status is available through the request variable (see MPI_Test() function)
     DO NOT change/update the variable buff until the message has been sent !!!

     MPI_Recv(void *buff, int count, MPI_Datatype type, 
     int source, int tag, int comm, MPI_Request *request )

     Receive a point-to-point message ASYNCHRONOUSLY
     The message MUST BE from the process source in the communication group comm AND the message MUST BE tagged with the tag value tag
     The message received will be stored at memory location buff which will have space to store count items of datatype type
     The MPI_Irecv() function will return IMMEDIATELY -
     The current RECEIVE status is available through the request variable (see MPI_Test() function)
     DO NOT use the variable buff until the message has been received !!! 
     */

  // Recv
  n_recv_req=0;
  for (int i=0; i<n_neighbors; i++)
  {
    int nb = f.nb[i];
    if(nb==-1) continue;
    MPI_Irecv( (void*)f.gr[ i ]->data, f.gr[ i ]->get_num()*sizeof(T), MPI_BYTE, 
        nb, (n_neighbors-1)-i, MPI_COMM_WORLD, &recv_requests[ n_recv_req++ ]);
  }

  // Send
  n_send_req=0;
  for (int i=0; i<n_neighbors; i++)
  {
    int nb = f.nb[i];
    if( nb ==-1) continue;
    MPI_Isend( (void*)f.gs[ i ]->data, f.gs[ i ]->get_num()*sizeof(T), MPI_BYTE,
        nb, i, MPI_COMM_WORLD, &send_requests[ n_send_req++ ]);
  }
}

  template<typename T>
void print_by_rank(int n_ranks, int my_rank, Function2D<T> &u)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int message=0;
  if (my_rank == 0)
  {
    printf("[%d] \n",my_rank);
    print_matrix(u.m, u.n, u.l, u.data);
    // u.print_ghost_cells(1, my_rank);
    // u.dump_data_to_file("out", my_rank);
    MPI_Send(&message, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
  } else {
    int buffer;
    MPI_Status status;
    MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_INT, &buffer);
    if (buffer == 1)
    {
      printf("[%d] \n",my_rank);
      print_matrix(u.m, u.n, u.l, u.data);
      // u.print_ghost_cells(1, my_rank);
      // u.dump_data_to_file("out", my_rank);
      MPI_Recv(&message, buffer, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
      if (my_rank + 1 != n_ranks)
      {
        MPI_Send(&message, 1, MPI_INT, ++my_rank, 0, MPI_COMM_WORLD);
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
}
#endif




