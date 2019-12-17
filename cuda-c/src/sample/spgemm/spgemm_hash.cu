#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

#include <math.h>
#include <algorithm>

#include <cuda.h>
#include <helper_cuda.h>

#include <nsparse.h>

void spgemm_csr(sfCSR *a, sfCSR *b, sfCSR *c)
{

    int i;
  
    long long int flop_count;
    cudaEvent_t event[2];
    float msec, ave_msec, flops;
  
    for (i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }
  
    /* Memcpy A and B from Host to Device */
    csr_memcpy(a);
    csr_memcpy(b);
  
    /* Count flop of SpGEMM computation */
    get_spgemm_flop(a, b, a->M, &flop_count);

    /* Execution of SpGEMM on Device */
    ave_msec = 0;
    for (i = 0; i < SPGEMM_TRI_NUM; i++) {
        if (i > 0) {
            release_csr(*c);
        }
        cudaEventRecord(event[0], 0);
        spgemm_kernel_hash(a, b, c);
        cudaEventRecord(event[1], 0);
        cudaThreadSynchronize();
        cudaEventElapsedTime(&msec, event[0], event[1]);

        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= SPGEMM_TRI_NUM - 1;
  
    flops = (float)(flop_count) / 1000 / 1000 / ave_msec;
  
    printf("SpGEMM using CSR format (Hash-based): %s, %f[GFLOPS], %f[ms]\n", a->matrix_name, flops, ave_msec);

    csr_memcpyDtH(c);
    release_csr(*c);
    
    /* Check answer */
#ifdef sfDEBUG
    sfCSR ans;
    // spgemm_cu_csr(a, b, &ans);

    printf("(nnz of A): %d =>\n(Num of intermediate products): %ld =>\n(nnz of C): %d\n", a->nnz, flop_count / 2, c->nnz);
    ans = *c;
    check_spgemm_answer(*c, ans);

    release_cpu_csr(ans);
#endif
  
    release_csr(*a);
    release_csr(*b);
    for (i = 0; i < 2; i++) {
        cudaEventDestroy(event[i]);
    }

}

__global__ void printSMTH() {
    printf("GRSIZE FROM MAIN %d", device_grammar_size);
}

/* Main Function */
int main(int argc, char **argv)
{
    sfCSR mat_a, mat_b, mat_c;
  
    /* Set CSR reading from MM file */
    int grammar_size = 3;
    unsigned short * grammar_body = (unsigned short *)calloc(grammar_size, sizeof(unsigned short));
    grammar_body[0] = 0x1;
    grammar_body[1] = 0x2;
    grammar_body[2] = 0x4;
    unsigned int * grammar_tail = (unsigned int *)calloc(grammar_size, sizeof(unsigned int));
    grammar_tail[0] = 0x00110011;
    grammar_tail[1] = 0x00100010;
    grammar_tail[2] = 0x00000010;
    printSMTH<<<1,1>>>();
    unsigned short * global_device_grammar_body = 0;
    unsigned int * global_device_grammar_tail = 0;

    cudaMalloc((void**)&global_device_grammar_body, grammar_size * sizeof(unsigned short));
    cudaMalloc((void**)&global_device_grammar_tail, grammar_size * sizeof(unsigned int));

    cudaMemcpy(global_device_grammar_body, grammar_body, grammar_size * sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaMemcpy(global_device_grammar_tail, grammar_tail, grammar_size * sizeof(unsigned int), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(device_grammar_body, global_device_grammar_body, grammar_size * sizeof(unsigned short));
    cudaMemcpyToSymbol(device_grammar_tail, global_device_grammar_tail, grammar_size * sizeof(unsigned int));
    cudaError_t result = cudaMemcpyToSymbol(device_grammar_size, &grammar_size, sizeof(int));
    if (result != cudaSuccess) {
        printf("PROBLEM: %s\n", cudaGetErrorString(result));
    }
    printSMTH<<<1,1>>>();
    cudaDeviceSynchronize();
    init_csr_matrix_from_file(&mat_a, argv[1]);
    init_csr_matrix_from_file(&mat_b, argv[1]);
  
    spgemm_csr(&mat_a, &mat_b, &mat_c);

    release_cpu_csr(mat_a);
    release_cpu_csr(mat_b);
    release_cpu_csr(mat_c);
    
    return 0;
}
