/*
 * rng.c -- deterministic LCG + Box-Muller sampling (see rng.h).
 */
#include "rng.h"

#include <math.h>

#define RNG64_MULT   6364136223846793005ULL
#define RNG64_INC    1442695040888963407ULL
#define RNG53_2EXP53 9007199254740992.0 /* 2^53 */

void rng64_seed(rng64 *g, uint64_t seed)
{
    g->state = seed;
    g->spare = 0.0;
    g->has_spare = 0;
    /* discard the first step so that small seeds do not start correlated */
    g->state = g->state * RNG64_MULT + RNG64_INC;
}

uint64_t rng64_next(rng64 *g)
{
    g->state = g->state * RNG64_MULT + RNG64_INC;
    return g->state;
}

double rng64_uniform(rng64 *g)
{
    uint64_t s = rng64_next(g);
    return (double)(s >> 11) / RNG53_2EXP53; /* 53 high bits -> [0,1) */
}

double rng64_normal(rng64 *g)
{
    double u1, u2, r;

    if (g->has_spare) {
        g->has_spare = 0;
        return g->spare;
    }
    u1 = rng64_uniform(g);
    u2 = rng64_uniform(g);
    if (u1 < 1.0 / RNG53_2EXP53)
        u1 = 1.0 / RNG53_2EXP53; /* keep log finite (2^-53 floor) */
    r = sqrt(-2.0 * log(u1));
    g->spare = r * sin(2.0 * 3.14159265358979323846 * u2);
    g->has_spare = 1;
    return r * cos(2.0 * 3.14159265358979323846 * u2);
}
