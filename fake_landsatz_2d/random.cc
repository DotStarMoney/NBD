#include "random.h"

#include <stdint.h>

#include <functional>
#include <limits>
#include <thread>

namespace nbd {
namespace util {

struct XorShiftP {
  XorShiftP() {
    srnd(std::hash<std::thread::id>{}(std::this_thread::get_id()));
  }
  uint64_t state[2];

  uint64_t Step() {
    uint64_t x = state[0];
    const uint64_t y = state[1];
    state[0] = y;

    x ^= x << 23;
    x ^= x >> 17;
    x ^= y ^ (y >> 26);

    state[1] = x;

    return x + y;
  }

  void Seed(uint64_t s) {
    state[0] = s;
    state[1] = 0x5ea34222ef71888b;
    for (uint32_t i = 0; i < 16; ++i) rnd();
  }
};
thread_local XorShiftP prng;

uint64_t rnd() { return prng.Step(); }

double rndd() {
  return static_cast<double>(rnd()) / std::numeric_limits<uint64_t>::max();
}

void srnd(uint64_t s) { prng.Seed(s); }

bool TrueWithChance(double prob) { return rndd() < prob; }

}  // namespace util
}  // namespace nbd
