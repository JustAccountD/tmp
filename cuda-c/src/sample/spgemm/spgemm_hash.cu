#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

#include <math.h>
#include <algorithm>

#include <cuda.h>
#include <helper_cuda.h>

#include <nsparse.h>


// C = A | B
// sz - amount of rows (we sum square matrix)
__global__ void sumSparse(int sz, int * rrzA, int * valA, int * colA, int * rrzB, int * valB, int * colB, int * rrzC, int * valC, int * colC)
{
    int colAcnt = 0;
    int colBcnt = 0;
    int colCcnt = 0;
    int i;
    int newrrz = 0;
    rrzC[0] = 0;
    for (i = 0; i < sz; i++) {

        printf("In start of while: %d %d\n", colAcnt, colBcnt);
        while (colAcnt < rrzA[i + 1] || colBcnt < rrzB[i + 1]) {
            newrrz++;


            // if both matrix are in game
            if (colAcnt < rrzA[i + 1] && colBcnt < rrzB[i + 1]) {
                printf("Col nums: %d %d\n", colA[colAcnt], colB[colBcnt]);
                if (colA[colAcnt] <= colB[colBcnt]) {
                    colC[colCcnt] = colA[colAcnt];
                    if (colA[colAcnt] == colB[colBcnt]) {
                        valC[colCcnt] = valA[colAcnt] | valB[colBcnt];
                        colBcnt++;
                    } else {
                        valC[colCcnt] = valA[colAcnt];
                    }
                    colCcnt++;
                    colAcnt++;
                } else {
                    colC[colCcnt] = colB[colBcnt];
                    valC[colCcnt] = valB[colBcnt];
                    colCcnt++;
                    colBcnt++;
                }
            } else if (colAcnt < rrzA[i + 1]) {
                colC[colCcnt] = colA[colAcnt];
                valC[colCcnt] = valA[colAcnt];
                colCcnt++;
                colAcnt++;
            } else {
                colC[colCcnt] = colB[colBcnt];
                valC[colCcnt] = valB[colBcnt];
                colCcnt++;
                colBcnt++;
            }
        }

        rrzC[i + 1] = newrrz;
    }
}



void spgemm_csr(sfCSR *a, sfCSR *b, sfCSR *c, int grSize, unsigned short int * grBody, unsigned int * grTail)
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
        spgemm_kernel_hash(a, b, c, grSize, grBody, grTail);
        checkCudaErrors(cudaMalloc((void **)&(b->d_col), sizeof(int) * (a->nnz + c->nnz)));
        checkCudaErrors(cudaMalloc((void **)&(b->d_val), sizeof(real) * (a->nnz + c->nnz)));
        sumSparse<<<1, 1>>>(a->M, a->d_rpt, a->d_val, a->d_col, c->d_rpt, c->d_val, c->d_col, b->d_rpt, b->d_val, b->d_col);
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

    // for test
    c = b;


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




/* Main Function */
int main(int argc, char **argv)
{
    sfCSR mat_a, mat_b, mat_c;
  
    /* Set CSR reading from MM file */
    int grammar_size = 3;
    unsigned short * grammar_body = (unsigned short *)calloc(grammar_size, sizeof(unsigned short));
    grammar_body[0] = 0x4;
    grammar_body[1] = 0x8;
    grammar_body[2] = 0x4;
    unsigned int * grammar_tail = (unsigned int *)calloc(grammar_size, sizeof(unsigned int));
    grammar_tail[0] = 0x00030003;
    grammar_tail[1] = 0x00070007;
    grammar_tail[2] = 0x00000010;
    cudaDeviceSynchronize();
    init_csr_matrix_from_file(&mat_a, argv[1]);
    init_csr_matrix_from_file(&mat_b, argv[1]);
  
    spgemm_csr(&mat_a, &mat_b, &mat_c, grammar_size, grammar_body, grammar_tail);

    release_cpu_csr(mat_a);
    release_cpu_csr(mat_b);
    release_cpu_csr(mat_c);
    
    return 0;
}
