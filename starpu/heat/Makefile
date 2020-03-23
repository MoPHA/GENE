STARPU_VERSION=1.3

# LD_LIBRARY_PATH=${HOME}/starpu/include/starpu/${STARPU_VERSION}
# PKG_CONFIG_PATH= ${HOME}/starpu/lib
# SPU_FLAGS=-lstarpumpi-${STARPU_VERSION} -lstarpu-${STARPU_VERSION} -lhwloc -lm
# vector_spu:
# 	gfortran -c vector_spu.f90 fstarpu_mod.f90
# 	gfortran -I ${LD_LIBRARY_PATH} vector_spu.o  fstarpu_mod.o  -o vector.spu -L ${PKG_CONFIG_PATH} ${SPU_FLAGS}
# clean:
# 	# rm ${FILENAME}
# 	rm vector_spu.o fstarpu_mod.o

#export STARPU_GENERATE_TRACE=1;
#SPU_FLAGS=-lstarpumpi-${STARPU_VERSION} -lstarpu-${STARPU_VERSION} -lhwloc -lm
#MPICC=g++-9

# ---------------------------------------------------------------------------- 
#  MPI only
# ---------------------------------------------------------------------------- 
FILENAME=heat_conv
EXT=cpp
MPICC=mpic++
CFLAGS = $$(pkg-config --cflags starpu-${STARPU_VERSION}) 
LDLIBS = $$(pkg-config --libs   starpu-${STARPU_VERSION}) -lstarpumpi-${STARPU_VERSION}
$(FILENAME):
	${MPICC} ${CFLAGS} ${FILENAME}.${EXT}  -o ${FILENAME} ${LDLIBS} 
clean:
	rm ${FILENAME}

# # ---------------------------------------------------------------------------- 
# #  CUDA and C++ only
# # ---------------------------------------------------------------------------- 
# CFLAGS = $$(pkg-config --cflags starpu-${STARPU_VERSION}) 
# LDLIBS = $$(pkg-config --libs   starpu-${STARPU_VERSION})
# CC=g++
# NVCC=nvcc
# NVCC_CFLAGS=-ccbin g++-7 ${CFLAGS}
# # NVCC_CFLAGS=-ccbin ${CC} -gencode arch=compute_50,code=sm_50  ${CFLAGS}
# NVCC_LDLIBS=-L/usr/local/cuda/lib64 -lcuda -lcudart -lstdc++
# convolution_spu_gpu: convolution_spu_gpu.o convolution_cuda_kernel.o 
# 	${CC} ${CFLAGS} -o convolution_spu_gpu convolution_spu_gpu.o convolution_cuda_kernel.o ${LDLIBS} ${NVCC_LDLIBS}
# convolution_spu_gpu.o:
# 	${CC} ${CFLAGS} -c convolution_spu_gpu.cpp -o convolution_spu_gpu.o
# convolution_cuda_kernel.o:
# 	${NVCC} ${NVCC_CFLAGS}  -c convolution_cuda_kernel.cu -o convolution_cuda_kernel.o 
# clean:
# 	rm -f convolution_spu_gpu *.o


# FILENAME=heat_spu
# EXT=cpp
# MPICC=mpic++
# CFLAGS = $$(pkg-config --cflags starpu-${STARPU_VERSION}) 
# LDLIBS = $$(pkg-config --libs   starpu-${STARPU_VERSION})
# CC=g++
# NVCC=nvcc
# NVCC_CFLAGS=-ccbin g++-7 ${CFLAGS}
# # NVCC_CFLAGS=-ccbin ${CC} -gencode arch=compute_50,code=sm_50  ${CFLAGS}
# NVCC_LDLIBS=-L/usr/local/cuda/lib64 -lcuda -lcudart -lstdc++
# $(FILENAME): ${FILENAME}.o convolution_cuda_kernel.o 
# 	${MPICC} -o ${FILENAME} ${FILENAME}.o convolution_cuda_kernel.o  ${LDLIBS} ${NVCC_LDLIBS} -lstarpumpi-${STARPU_VERSION}
# ${FILENAME}.o:
# 	${MPICC} ${CFLAGS} -c ${FILENAME}.cpp -o ${FILENAME}.o
# convolution_cuda_kernel.o:
# 	${NVCC} ${NVCC_CFLAGS}  -c convolution_cuda_kernel.cu -o convolution_cuda_kernel.o 
# clean:
# 	rm ${FILENAME} *.o


