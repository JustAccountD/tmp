/*
 * Inline PTX
 */
__device__ __inline__ real ld_gbl_val(const real *val)
{
    real return_value;

#ifdef FLOAT
    asm("ld.global.cv.f32 %0, [%1];" : "=f"(return_value) : "l"(val));
#else
    asm("ld.global.cv.f64 %0, [%1];" : "=d"(return_value) : "l"(val));
#endif
  
    return return_value;
}

__device__ __inline__ float2 ld_gbl_float2(const float2 *val)
{
    float2 return_value;

    asm("ld.global.cv.v2.f32 {%0, %1}, [%2];" : "=f"(return_value.x), "=f"(return_value.y) : "l"(val));
    return return_value;
}

__device__ __inline__ float4 ld_gbl_float4(const float4 *val)
{
    float4 return_value;

    asm("ld.global.cv.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(return_value.x), "=f"(return_value.y), "=f"(return_value.z), "=f"(return_value.w) : "l"(val));
    return return_value;
}

__device__ __inline__ short ld_gbl_row(const short *row)
{
    short return_value;
    asm("ld.global.cv.u16 %0, [%1];" : "=h"(return_value) : "l"(row));
    return return_value;
}

__device__ __inline__ int ld_gbl_col(const int *col)
{
    int return_value;
    asm("ld.global.cv.s32 %0, [%1];" : "=r"(return_value) : "l"(col));
    return return_value;
}

__device__ __inline__ short ld_gbl_short(const short *col)
{
    short return_value;
    asm("ld.global.cv.u16 %0, [%1];" : "=h"(return_value) : "l"(col));
    return return_value;
}

__device__ __inline__ unsigned short ld_gbl_ushort(const unsigned short *col)
{
    unsigned short return_value;
    asm("ld.global.cv.u16 %0, [%1];" : "=h"(return_value) : "l"(col));
    return return_value;
}

__device__ __inline__ unsigned char ld_gbl_uchar(const unsigned char *row)
{
    short return_value;
    asm("ld.global.cv.u8 %0, [%1];" : "=h"(return_value) : "l"(row));
    return (unsigned char)return_value;
}

__device__ __inline__ void st_gbl_ans(const real *ans_gpu, real answer)
{
#ifdef FLOAT
    asm("st.global.cs.f32 [%0], %1;" :: "l"(ans_gpu) , "f"(answer));
#else
    asm("st.global.cs.f64 [%0], %1;" :: "l"(ans_gpu) , "d"(answer));
#endif

}

#define INT16 1

__device__ __inline__ real ld_gbl_real(const real *val) {

  real return_value;

#ifdef FLOAT
#ifdef INT16
  asm("ld.global.cv.u16 %0, [%1];" : "=h"(return_value) : "l"(val));
#elif defined INT32
  asm("ld.global.cv.u32 %0, [%1];" : "=r"(return_value) : "l"(val));
#endif
#else
  asm("ld.global.cv.f64 %0, [%1];" : "=d"(return_value) : "l"(val));
#endif
  
  return return_value;
}

__device__ __inline__ int ld_gbl_int32(const int *col) {
  int return_value;
  asm("ld.global.cv.s32 %0, [%1];" : "=r"(return_value) : "l"(col));
  return return_value;
}

__device__ unsigned short atomicAddShort(unsigned short* address, unsigned short val)
{
    unsigned int *base_address = (unsigned int *) ((char *)address - ((size_t)address & 2));	//tera's revised version (showtopic=201975)
    unsigned int long_val = ((size_t)address & 2) ? ((unsigned int)val << 16) : (unsigned short)val;
    unsigned int long_old = atomicAdd(base_address, long_val);
    if ((size_t)address & 2) {
        return (unsigned short)(long_old >> 16);
    } else {
        unsigned int overflow = ((long_old & 0xffff) + long_val) & 0xffff0000;
        if (overflow)
            atomicSub(base_address, overflow);
        return (unsigned short)(long_old & 0xffff);
    }
}



__device__ __inline__ void atomic_fadd(real *adr, real val)
{
//#if __CUDA_ARCH__ >= 600
//    atomicAdd(adr, val);
//    printf("600ARCH");
//#else
#ifdef FLOAT
#ifdef INT16
    unsigned short int *address_ull = (unsigned short int *)(adr);
    printf("BEFORE: %d for %p\n", *address_ull, address_ull);
    printf("VAL: %d for %p\n", val, address_ull);
    real input = val;
    atomicAddShort(adr, val);
    printf("RES: %d for %p\n", *address_ull, address_ull);
#elif defined INT32
    unsigned int *address_ull = (unsigned int *)(adr);
    printf("BEFORE: %d for %p\n", *address_ull, address_ull);
    printf("VAL: %d for %p\n", val, address_ull);
    unsigned int old_val = *address_ull;
    unsigned int assumed;
    real input = val;
    do {
        assumed = old_val;
        old_val = atomicCAS(address_ull, assumed, input + assumed);
    } while (assumed != old_val);
    printf("RES: %d for %p\n", *address_ull, address_ull);
#endif
#elif defined DOUBLE
    unsigned long long int *address_ull = (unsigned long long int *)(adr);
    unsigned long long int old_val = *address_ull;
    unsigned long long int assumed;
    real input = val;
    do {
        assumed = old_val;
        old_val = atomicCAS(address_ull, assumed, __double_as_longlong(input + __longlong_as_double(assumed)));
    } while (assumed != old_val);
#endif
//#endif
}

