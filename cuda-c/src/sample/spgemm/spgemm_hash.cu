#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

#include <math.h>
#include <algorithm>

#include <cuda.h>
#include <helper_cuda.h>

#include <nsparse.h>

#ifdef FLOAT
void csr_copy(sfCSR * src, sfCSR * dst) {
    release_csr(*dst);
    dst->M = src->M;
    dst->N = src->N;
    dst->nnz = src->nnz;
    dst->nnz_max = src->nnz_max;

    checkCudaErrors(cudaMalloc((void **)&(dst->d_rpt), sizeof(int) * (dst->M + 1)));
    checkCudaErrors(cudaMalloc((void **)&(dst->d_col), sizeof(int) * dst->nnz));
    checkCudaErrors(cudaMalloc((void **)&(dst->d_val), sizeof(real) * dst->nnz));

    checkCudaErrors(cudaMemcpy(dst->d_rpt, src->d_rpt, sizeof(int) * (src->M + 1), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(dst->d_col, src->d_col, sizeof(int) * src->nnz, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(dst->d_val, src->d_val, sizeof(real) * src->nnz, cudaMemcpyDeviceToDevice));
}

__device__ int flagNoChange = true;

// C = A | B and check if C == A (if they are equal flagNoChange will be false)
// sz - amount of rows (we sum square matrix)
__global__ void sumSparse(int sz, int * rrzA, real * valA, int * colA, int * rrzB, real * valB, int * colB, int * rrzC, real * valC, int * colC)
{
    flagNoChange = true;
    int colAcnt = 0;
    int colBcnt = 0;
    int colCcnt = 0;
    int i;
    int newrrz = 0;
    rrzC[0] = 0;
    for (i = 0; i < sz; i++) {

        //printf("In start of while: %d %d\n", colAcnt, colBcnt);
        while (colAcnt < rrzA[i + 1] || colBcnt < rrzB[i + 1]) {

            if (colAcnt < rrzA[i + 1] && valA[colAcnt] == 0) {
                colAcnt++;
                continue;
            }

            if (colBcnt < rrzB[i + 1] && valB[colBcnt] == 0) {
                colBcnt++;
                continue;
            }

            newrrz++;

            // if both matrix are in game
            if (colAcnt < rrzA[i + 1] && colBcnt < rrzB[i + 1]) {
                if (colA[colAcnt] <= colB[colBcnt]) {
                    colC[colCcnt] = colA[colAcnt];
                    if (colA[colAcnt] == colB[colBcnt]) {
                        valC[colCcnt] = valA[colAcnt] | valB[colBcnt];
                        if (valC[colCcnt] != valA[colAcnt]) {
                            flagNoChange = -valB[colAcnt];
                        }
                        colBcnt++;
                    } else {
                        valC[colCcnt] = valA[colAcnt];
                    }
                    colCcnt++;
                    colAcnt++;
                } else {
                    colC[colCcnt] = colB[colBcnt];
                    valC[colCcnt] = valB[colBcnt];
                    flagNoChange = 555;
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
                flagNoChange = 444 + valB[colBcnt];
                colCcnt++;
                colBcnt++;
            }
        }

        rrzC[i + 1] = newrrz;
    }
}
#endif


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
#ifdef FLOAT
        int noChange = 0;
        bool first = true;
        int u = 0;
        while (/*!noChange*/true) {
            u++;
            if (u > 4) {
                break;
            }
            printf("Ready for mult\n");
            if (first) {
                first = false;
#endif
                spgemm_kernel_hash(a, b, c, grSize, grBody, grTail, true);
#ifdef FLOAT
            }
            else {
                release_csr(*c);
                spgemm_kernel_hash(a, b, c, grSize, grBody, grTail, false);
            }

            printf("Success mult!!\n");
            cudaFree(b->d_col);
            cudaFree(b->d_val);
            checkCudaErrors(cudaMalloc((void **)&(b->d_col), sizeof(int) * (a->nnz + c->nnz)));
            checkCudaErrors(cudaMalloc((void **)&(b->d_val), sizeof(real) * (a->nnz + c->nnz)));
            sumSparse<<<1, 1>>>(a->M, a->d_rpt, a->d_val, a->d_col, c->d_rpt, c->d_val, c->d_col, b->d_rpt, b->d_val, b->d_col);
            csr_copy(b, a);
            csr_copy(a, b);
            csr_memcpyDtH(b);
            printf("NNZ of sum: %d RPT last of sum: %d\n", b->nnz, b->rpt[4]);
            cudaError_t result = cudaGetLastError();
            if (result != cudaSuccess) {
                printf("PROBLEM1: %s\n", cudaGetErrorString(result));
            }
            cudaMemcpyFromSymbol(&noChange, flagNoChange, sizeof(int), 0, cudaMemcpyDeviceToHost);
            printf("FLAG: %d\n", noChange);
            result = cudaGetLastError();
            if (result != cudaSuccess) {
                printf("PROBLEM2: %s\n", cudaGetErrorString(result));
            }
            cudaThreadSynchronize();
        }
#endif
        cudaEventRecord(event[1], 0);
        cudaThreadSynchronize();
        cudaEventElapsedTime(&msec, event[0], event[1]);

#ifndef FLOAT
        if (i > 0) {
#endif
            ave_msec += msec;
#ifndef FLOAT
        }
#endif
    }
#ifndef FLOAT
    ave_msec /= SPGEMM_TRI_NUM - 1;
#endif
  
    flops = (float)(flop_count) / 1000 / 1000 / ave_msec;
  
    printf("SpGEMM using CSR format (Hash-based): %s, %f[GFLOPS], %f[ms]\n", a->matrix_name, flops, ave_msec);

#ifdef FLOAT
    c = b;
#endif


    csr_memcpyDtH(c);
#ifndef FLOAT
    release_csr(*c);
#endif

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
    int grammar_size = 2;
    unsigned short * grammar_body = (unsigned short *)calloc(grammar_size, sizeof(unsigned short));
//    grammar_body[0] = 0x1;
    grammar_body[0] = 0x4;
//    grammar_body[1] = 0x1;
    grammar_body[1] = 0x8;
//    grammar_body[2] = 0x1;
//    grammar_body[3] = 0x1;
//    grammar_body[4] = 0x4;
//    grammar_body[5] = 0x10;
    unsigned int * grammar_tail = (unsigned int *)calloc(grammar_size, sizeof(unsigned int));
//    grammar_tail[0] = 0x00020004;
//    grammar_tail[1] = 0x00080010;
    grammar_tail[0] = 0x00030003;
    grammar_tail[1] = 0x00040004;
//    grammar_tail[2] = 0x00020020;
//    grammar_tail[3] = 0x00080040;
//    grammar_tail[4] = 0x00010020;
//    grammar_tail[5] = 0x00010040;
    cudaDeviceSynchronize();
    init_csr_matrix_from_file(&mat_a, argv[1]);
    init_csr_matrix_from_file(&mat_b, argv[1]);
  
    spgemm_csr(&mat_a, &mat_b, &mat_c, grammar_size, grammar_body, grammar_tail);

    release_cpu_csr(mat_a);
    release_cpu_csr(mat_b);
#ifndef FLOAT
    release_cpu_csr(mat_c);
#endif
    
    return 0;
}
