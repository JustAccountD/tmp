CXX = g++
NVCC = nvcc

CFLAGS = -O3 -g -std=c++11
CFLAGS += -L. ${REAL} -lm
LDFLAGS = ${CFLAGS}

# for Device Code
CUDA_PATH = /usr/local/cuda
LDFLAGS += -L${CUDA_PATH}/lib64
LDFLAGS += -lm -arch=sm_60 -lcudart -lcusparse
INCLUDE = -I./inc
INCLUDE += -I${CUDA_PATH}/include
INCLUDE += -I${CUDA_PATH}/samples/common/inc

BIN = ./bin
SRC = ./src
OBJ = ./obj

OBJ_SUF = .o
OS_SUF = .s.o
OD_SUF = .d.o

OBJS_LIB = $(OBJ)/nsparse.o
OBJS_CU_CSR = $(OBJS_LIB) $(OBJ)/sample/spmv/spmv_cu_csr.o
OBJS_AMB = $(OBJS_LIB) $(OBJ)/conversion/convert_amb.o $(OBJ)/kernel/kernel_spmv_amb.o $(OBJ)/sample/spmv/spmv_amb.o
OBJS_SPGEMM_CU_CSR = $(OBJS_LIB) $(OBJ)/kernel/kernel_spgemm_cu_csr.o $(OBJ)/sample/spgemm/spgemm_cu_csr.o
OBJS_SPGEMM_HASH_S = $(OBJS_LIB) $(OBJ)/kernel/kernel_spgemm_cu_csr.o $(OBJ)/kernel/kernel_spgemm_hash_s.o $(OBJ)/sample/spgemm/spgemm_hash.o
OBJS_SPGEMM_HASH_D = $(OBJS_LIB) $(OBJ)/kernel/kernel_spgemm_cu_csr.o $(OBJ)/kernel/kernel_spgemm_hash_d.o $(OBJ)/sample/spgemm/spgemm_hash.o
OBJS_SPGEMM_HASH_KERNEL_GEN = $(OBJ)/kernel/spgemm_hash_kernel_gen.o

spgemm_hash :
	make spgemm_hash_s

spgemm_hash_s : $(OBJS_SPGEMM_HASH_S:$(OBJ_SUF)=$(OS_SUF))
	mkdir -p $(BIN)
	$(NVCC) -o $(BIN)/$@ $(OBJS_SPGEMM_HASH_S:$(OBJ_SUF)=$(OS_SUF)) $(LDFLAGS) $(INCLUDE)

$(OBJ)/%$(OS_SUF) : $(SRC)/%.c
	mkdir -p $(dir $@)
	$(CXX) -c -DFLOAT $(CFLAGS) $(INCLUDE) -o $@ $<

$(OBJ)/%$(OD_SUF) : $(SRC)/%.c
	mkdir -p $(dir $@)
	$(CXX) -c -DDOUBLE $(CFLAGS) $(INCLUDE) -o $@ $<

$(OBJ)/%$(OS_SUF) : $(SRC)/%.cu
	mkdir -p $(dir $@)
	$(NVCC) -c -DFLOAT $(LDFLAGS) $(INCLUDE) -o $@ $<

$(OBJ)/%$(OD_SUF) : $(SRC)/%.cu
	mkdir -p $(dir $@)
	$(NVCC) -c -DDOUBLE $(LDFLAGS) $(INCLUDE) -o $@ $<

clean :
	rm -rf $(BIN)/*
	rm -rf $(OBJ)/*
