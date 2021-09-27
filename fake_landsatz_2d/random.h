#ifndef NBD_UTIL_RANDOM_H_
#define NBD_UTIL_RANDOM_H_

#include <stdint.h>

namespace nbd {
namespace util {

// Produces a pseudo-random 64bit int. Repeated calls to this PRNG will almost
// always produce a unique stream of values per-thread.
uint64_t rnd();

// Produces a pseudo-random double in the range [0, 1). Uses rnd().
double rndd();

// Provide a specific seed to the PRNG in the current thread. A given seed will
// always produce the same stream of random values.
void srnd(uint64_t s);

// Returns true with a given probability. If prob >= 1, always returns true. If
// prob <= 0, always returns false. Uses rndd().
bool TrueWithChance(double prob);

}  // namespace util
}  // namespace nbd

#endif  // NBD_UTIL_RANDOM_H_
