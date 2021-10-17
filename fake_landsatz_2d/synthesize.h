#ifndef NBD_SYNTHESIZE_H_
#define NBD_SYNTHESIZE_H_

#include <stdint.h>

#include <memory>

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
  float break_pattern_point_plot_prob;
  float base_harmonic_freq_day_radius;
  float base_harmonic_max_amplitude;
  float base_harmonic_amplitude_sat;
  float fast_harmonic_freq_min_days;
  float fast_harmonic_freq_max_days;
  float fast_harmonic_max_amplitude;
  float fast_harmonic_amplitude_sat;
  float slow_harmonic_prob;
  float slow_harmonic_freq_min_years;
  float slow_harmonic_freq_max_years;
  float slow_harmonic_max_amplitude;
  float slow_harmonic_amplitude_sat;
  float harmonic_tension_max;
};

constexpr float kYearDays = 365;

const SynthesizeConfig kDefaultConfig = {
    .depth = 4096,
    .height = 5,
    .width = 5,
    .event_prob = 0.0015,
    .unmask_prob_min = 0.08,
    .unmask_prob_max = 0.12,
    .event_min_period = 365,
    .raw_spine_prob = 0.01,
    .break_pattern_multiplier = 6,
    .break_pattern_shapes_n = 100,
    .break_pattern_point_plot_prob = 0.8,
    .base_harmonic_freq_day_radius = 30,
    .base_harmonic_max_amplitude = 0.2,
    .base_harmonic_amplitude_sat = 1,
    .fast_harmonic_freq_min_days = 60,
    .fast_harmonic_freq_max_days = kYearDays,
    .fast_harmonic_max_amplitude = 0.2,
    .fast_harmonic_amplitude_sat = 2,
    .slow_harmonic_prob = 0.2,
    .slow_harmonic_freq_min_years = 2,
    .slow_harmonic_freq_max_years = 5,
    .slow_harmonic_max_amplitude = 0.1,
    .slow_harmonic_amplitude_sat = 2,
    .harmonic_tension_max = 8};

SynthesizeResult Synthesize(const SynthesizeConfig& config = kDefaultConfig);

}  // namespace nbd

#endif  // NBD_SYNTHESIZE_H_
