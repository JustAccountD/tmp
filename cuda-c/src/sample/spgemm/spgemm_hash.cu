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
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <fstream>
using namespace std;

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
__device__ int nnzSum = 0;

// C = A | B and check if C == A (if they are equal flagNoChange will be false)
// sz - amount of rows (we sum square matrix)
__global__ void sumSparse(int sz, int * rptA, real * valA, int * colA, int * rptB, real * valB, int * colB, int * rptC, real * valC, int * colC)
{
    flagNoChange = true;
    int colAcnt = 0;
    int colBcnt = 0;
    int colCcnt = 0;
    int i;
    int newrpt = 0;
    rptC[0] = 0;
    for (i = 0; i < sz; i++) {

        //printf("In start of while: %d %d\n", colAcnt, colBcnt);
        while (colAcnt < rptA[i + 1] || colBcnt < rptB[i + 1]) {

            if (colAcnt < rptA[i + 1] && valA[colAcnt] == 0) {
                colAcnt++;
                continue;
            }

            if (colBcnt < rptB[i + 1] && valB[colBcnt] == 0) {
                colBcnt++;
                continue;
            }

            newrpt++;

            // if both matrix are in game
            if (colAcnt < rptA[i + 1] && colBcnt < rptB[i + 1]) {
                if (colA[colAcnt] <= colB[colBcnt]) {
                    colC[colCcnt] = colA[colAcnt];
                    if (colA[colAcnt] == colB[colBcnt]) {
                        valC[colCcnt] = valA[colAcnt] | valB[colBcnt];
                        if (valC[colCcnt] != valA[colAcnt]) {
                            flagNoChange = false;
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
                    flagNoChange = false;
                    colCcnt++;
                    colBcnt++;
                }
            } else if (colAcnt < rptA[i + 1]) {
                colC[colCcnt] = colA[colAcnt];
                valC[colCcnt] = valA[colAcnt];
                colCcnt++;
                colAcnt++;
            } else {
                colC[colCcnt] = colB[colBcnt];
                valC[colCcnt] = valB[colBcnt];
                flagNoChange = false;
                colCcnt++;
                colBcnt++;
            }
        }

        rptC[i + 1] = newrpt;
        nnzSum = newrpt;
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
        int nnzS = 0;
        int u = 0;
        while (!noChange) {
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
            cudaMemcpyFromSymbol(&nnzS, nnzSum, sizeof(int), 0, cudaMemcpyDeviceToHost);
            b->nnz = nnzS;
            csr_copy(b, a);
            csr_copy(a, b);

            //printf("NNZ of sum: %d RPT last of sum: %d\n", b->nnz, b->rpt[4]);
            cudaError_t result = cudaGetLastError();
            if (result != cudaSuccess) {
                printf("PROBLEM1: %s\n", cudaGetErrorString(result));
            }
            cudaMemcpyFromSymbol(&noChange, flagNoChange, sizeof(int), 0, cudaMemcpyDeviceToHost);
            //printf("FLAG: %d\n", noChange);
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
#ifdef FLOAT
    int t, sumAmount = 0;
    for (t = 0; t < c->nnz; t++) {
        if ((c->val[t] & 0x1) == 0x1) {
            sumAmount++;
        }
    }
    printf("SumAmount: %d\n", sumAmount);
#endif
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


unsigned char toBoolVector(unsigned int number) {
    return ((real)0x1) << number;
}

std::unordered_map<std::string, std::vector<int> > terminal_to_nonterminals;

int load_grammar(const std::string & grammar_filename, real * grammar_body, unsigned int * grammar_tail) {
    auto chomsky_stream = std::ifstream(grammar_filename, ifstream::in);

    std::string line, tmp;
    unsigned int nonterminals_count = 0;
    unsigned int vertices_count = 0;

    std::map<std::string, unsigned int> nonterminal_to_index;
    std::vector<unsigned int> epsilon_nonterminals;
    std::vector<std::pair<unsigned int, std::pair<unsigned int, unsigned int> > > rules;

    while (getline(chomsky_stream, line)) {
        vector <std::string> terms;
        istringstream iss(line);
        while (iss >> tmp) {
            terms.push_back(tmp);
        }
        if (!nonterminal_to_index.count(terms[0])) {
            nonterminal_to_index[terms[0]] = nonterminals_count++;
        }
        if (terms.size() == 1) {
            epsilon_nonterminals.push_back(nonterminal_to_index[terms[0]]);
        } else if (terms.size() == 2) {
            if (!terminal_to_nonterminals.count(terms[1])) {
                terminal_to_nonterminals[terms[1]] = {};
            }
            terminal_to_nonterminals[terms[1]].push_back(nonterminal_to_index[terms[0]]);
        } else if (terms.size() == 3) {
            if (!nonterminal_to_index.count(terms[1])) {
                nonterminal_to_index[terms[1]] = nonterminals_count++;
            }
            if (!nonterminal_to_index.count(terms[2])) {
                nonterminal_to_index[terms[2]] = nonterminals_count++;
            }
            rules.push_back(
                    {nonterminal_to_index[terms[0]], {nonterminal_to_index[terms[1]], nonterminal_to_index[terms[2]]}});
        }
    }
    chomsky_stream.close();



    for (size_t i = 0; i < rules.size(); i++) {
        grammar_body[i] = toBoolVector(rules[i].first);
        grammar_tail[i] = (((unsigned int)toBoolVector(rules[i].second.first)) << (sizeof(real) * 8)) | (unsigned int)toBoolVector(rules[i].second.second);
    }

    return rules.size();
}

void load_graph(const std::string & graph_filename, sfCSR * matrix) {
    std::vector<std::pair<std::string, std::pair<unsigned int, unsigned int> > > edges;
    unsigned int vertices_count = 0;

    auto graph_stream = std::ifstream(graph_filename, ifstream::in);
    unsigned int from, to;
    std::string terminal;
    while (graph_stream >> from >> terminal >> to) {
        edges.push_back({terminal, {from, to}});
        vertices_count = max(vertices_count, max(from, to) + 1);
    }
    graph_stream.close();

    matrix->nnz = 0;
    matrix->M = vertices_count;
    matrix->N = vertices_count;
    col_coo = (int *)malloc(sizeof(int) * edges.size());
    row_coo = (int *)malloc(sizeof(int) * edges.size());
    val_coo = (real *)malloc(sizeof(real) * edges.size());
    int i = 0;

    for (auto & edge : edges) {
        if (terminal_to_nonterminals.count(edge.first) == 0) {
            continue;
        }
        auto nonterminals = terminal_to_nonterminals.at(edge.first);
        unsigned short bool_vector = 0;
        for (auto nonterminal : nonterminals) {
            bool_vector |= toBoolVector(nonterminal);
        }

        row_coo[i] = edge.second.first;
        col_coo[i] = edge.second.second;
        val_coo[i] = bool_vector;
    }


    /* Count the number of non-zero in each row */
    nnz_num = (int *)malloc(sizeof(int) * matrix->M);
    for (i = 0; i < matrix->M; i++) {
        nnz_num[i] = 0;
    }
    for (i = 0; i < num; i++) {
        nnz_num[row_coo[i]]++;
    }

    // Store matrix in CSR format
    /* Allocation of rpt, col, val */
    rpt_ = (int *)malloc(sizeof(int) * (matrix->M + 1));
    col_ = (int *)malloc(sizeof(int) * matrix->nnz);
    val_ = (real *)malloc(sizeof(real) * matrix->nnz);

    offset = 0;
    matrix->nnz_max = 0;
    for (i = 0; i < matrix->M; i++) {
        rpt_[i] = offset; // looks like we have amount of not null in rows before this row
        offset += nnz_num[i];
        if(matrix->nnz_max < nnz_num[i]){
            matrix->nnz_max = nnz_num[i];
        }
    }
    rpt_[matrix->M] = offset; // amount of all not null

    int * each_row_index = (int *)malloc(sizeof(int) * matrix->M);
    for (i = 0; i < matrix->M; i++) {
        each_row_index[i] = 0;
    }

    for (i = 0; i < num; i++) {
        col_[rpt_[row_coo[i]] + each_row_index[row_coo[i]]] = col_coo[i];
        val_[rpt_[row_coo[i]] + each_row_index[row_coo[i]]++] = val_coo[i];
    }

    matrix->rpt = rpt_;
    matrix->col = col_;
    matrix->val = val_;

    free(nnz_num);
    free(row_coo);
    free(col_coo);
    free(val_coo);
    free(each_row_index);
}


/* Main Function */
int main(int argc, char **argv)
{
    sfCSR mat_a, mat_b, mat_c;
  
    /* Set CSR reading from MM file */
    int grammar_size = 6;
    real * grammar_body = (unsigned short *)calloc(grammar_size, sizeof(unsigned short));
    grammar_body[0] = 0x1;
//    grammar_body[0] = 0x4;
    grammar_body[1] = 0x1;
//    grammar_body[1] = 0x8;
    grammar_body[2] = 0x1;
    grammar_body[3] = 0x1;
    grammar_body[4] = 0x4;
    grammar_body[5] = 0x10;
    unsigned int * grammar_tail = (unsigned int *)calloc(grammar_size, sizeof(unsigned int));
    grammar_tail[0] = 0x00020004;
    grammar_tail[1] = 0x00080010;
//    grammar_tail[0] = 0x00030003;
//    grammar_tail[1] = 0x00040004;
    grammar_tail[2] = 0x00020020;
    grammar_tail[3] = 0x00080040;
    grammar_tail[4] = 0x00010020;
    grammar_tail[5] = 0x00010040;
    cudaDeviceSynchronize();
    //init_csr_matrix_from_file(&mat_a, argv[1]);
    //init_csr_matrix_from_file(&mat_b, argv[1]);

    grammar_size = load_grammar(argv[1], grammar_body, grammar_tail);
    load_graph(argv[2], &mat_a);
    load_graph(argv[2], &mat_b);
  
    spgemm_csr(&mat_a, &mat_b, &mat_c, grammar_size, grammar_body, grammar_tail);

    release_cpu_csr(mat_a);
    release_cpu_csr(mat_b);
#ifndef FLOAT
    release_cpu_csr(mat_c);
#endif
    
    return 0;
}
