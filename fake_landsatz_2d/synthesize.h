#ifndef NBD_SYNTHESIZE_H_
#define NBD_SYNTHESIZE_H_

#include <memory>
#include <stdint.h>

namespace nbd {

struct SynthesizeResult {
  std::unique_ptr<float> raw;
  std::unique_ptr<float> distance;
  std::unique_ptr<float> spine;
  std::unique_ptr<int8_t> mask;
};

struct SynthesizeConfig {
  int depth;
  int height;
  int width;
  float event_prob;
  float unmask_prob_min;
  float unmask_prob_max;
  int event_min_period;
  float raw_spine_prob;
  int break_pattern_multiplier;
  int break_pattern_shapes_n;
};

const SynthesizeConfig kDefaultConfig = {.depth = 4096,
                                         .height = 5,
                                         .width = 5,
                                         .event_prob = 0.0015,
                                         .unmask_prob_min = 0.08,
                                         .unmask_prob_max = 0.12,
                                         .event_min_period = 365,
                                         .raw_spine_prob = 0.01,
                                         .break_pattern_multiplier = 6,
                                         .break_pattern_shapes_n = 100};

SynthesizeResult Synthesize(const SynthesizeConfig& config = kDefaultConfig);

}  // namespace nbd

#endif  // NBD_SYNTHESIZE_H_
