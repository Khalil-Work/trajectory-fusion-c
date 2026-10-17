/*
 * rng.h -- deterministic 64-bit linear congruential generator with a
 * Box-Muller standard-normal transform. No external dependencies.
 *
 * The generator is the Knuth MMIX LCG
 *     s <- 6364136223846793005 * s + 1442695040888963407  (mod 2^64)
 * and a uniform deviate in [0,1) is taken from the 53 high bits of s.
 * The same seed produces the same stream on every platform that maps
 * uint64_t arithmetic to unsigned 64-bit two's complement (C99 requires
 * at least 64 bits; all mainstream platforms are exact here).
 */
#ifndef RNG_H
#define RNG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t state;
    double   spare;     /* cached second Box-Muller deviate      */
    int      has_spare; /* 1 if `spare` holds an unused deviate  */
} rng64;

void   rng64_seed(rng64 *g, uint64_t seed);
uint64_t rng64_next(rng64 *g);        /* raw LCG step */
double rng64_uniform(rng64 *g);       /* U[0,1)      */
double rng64_normal(rng64 *g);        /* N(0,1)      */

#ifdef __cplusplus
}
#endif

#endif /* RNG_H */
