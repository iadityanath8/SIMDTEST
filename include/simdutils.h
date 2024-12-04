#pragma once 
#include <immintrin.h>

/** Loading  */

#define LOADLD(v) _mm256_loadu_pd(v)
#define LOADFL(v) _mm256_loadu_ps(v)
#define LOADNUM(v) _mm256_loadu_si256(v)

/** types macro */
#define simflt __m256
#define simnum __m256i
#define simdou __m256d

/** operations */
#define ADDFLT(veca, vecb) _mm256_add_ps(veca, vecb)
#define ADDDOU(veca, vecb) _mm256_add_pd(veca, vecb)
#define ADDINT(veca, vecb) _mm256_add_epi32(veca, vecb)
#define ADDLNG(veca, vecb) _mm256_add_epi64(veca, vecb)

/** Store macros */

#define STOREDOU(a, b) _mm256_storeu_pd(a, b)
#define STOREFLT(a, b) _mm256_storeu_ps(a, b)
#define STORENUM(a, b) _mm256_storeu_si256(a, b)