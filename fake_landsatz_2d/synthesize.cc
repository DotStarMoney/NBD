#include "synthesize.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdint.h>

#include <memory>

#include "break_pattern.h"
#include "random.h"

namespace nbd {
namespace {
template <typename ElementType>
std::unique_ptr<ElementType> CreateArray(int length) {
  return std::unique_ptr<ElementType>(new ElementType[length]);
}

float RandomTensionMu(float tension_max) {
  float mu = util::rndd() * tension_max;
  if (mu == 0) mu = 1.0;
  return util::TrueWithChance(0.5) ? 1.0f / mu : mu;
}

struct Harmonic {
  float frequency;
  float phase;

  float amplitude;
  float amplitude_max;
  float amplitude_saturate;

  float saturate;
  float saturate_max;

  void GenerateAmplitudeConstants() {
    amplitude = powf(util::rndd(), amplitude_saturate) * amplitude_max;
    saturate = RandomTensionMu(saturate_max);
  }

  float sample(float x) const {
    float h = 0.5f + sinf(x * frequency * phase) * 0.5f;
    return amplitude * 2.0f * (powf(h, saturate) - 0.5f);
  }
};

float RandomFrequency(float days_low, float days_high) {
  return M_PI * 2.0f / (util::rndd() * (days_high - days_low) + days_low);
}

void GenerateTimeSeries(const SynthesizeConfig& config,
                        const std::vector<int>& events, float* raw, 
                        float* distance, float* spine, int stride) {
  int intervals = events.size() - 1;
  auto bounds = CreateArray<int>(intervals * 2);
  for (int i = 0; i < intervals; ++i) {
    bounds.get()[2 * i] = events[i];
    bounds.get()[2 * i + 1] = events[i + 1];
  }

  int harmonics_n = 2;
  Harmonic harmonics[3];
  harmonics[0] = {.frequency = RandomFrequency(
                      kYearDays - config.base_harmonic_freq_day_radius,
                      kYearDays + config.base_harmonic_freq_day_radius), 
                  .phase = M_PI * 2 * util::rndd(),
                  .amplitude_max = config.base_harmonic_max_amplitude,
                  .amplitude_saturate = config.base_harmonic_amplitude_sat,
                  .saturate_max = config.harmonic_tension_max};
  harmonics[1] = {.frequency = RandomFrequency(
                      config.fast_harmonic_freq_min_days,
                      config.fast_harmonic_freq_max_days), 
                  .phase = M_PI * 2 * util::rndd(),
                  .amplitude_max = config.fast_harmonic_max_amplitude,
                  .amplitude_saturate = config.fast_harmonic_amplitude_sat,
                  .saturate_max = config.harmonic_tension_max};
  
  if (util::TrueWithChance(config.slow_harmonic_prob)) {
    harmonics_n = 3;
    harmonics[2] = {.frequency = RandomFrequency(
                        config.slow_harmonic_freq_min_years,
                        config.slow_harmonic_freq_max_years), 
                    .phase = M_PI * 2 * util::rndd(),
                    .amplitude_max = config.slow_harmonic_max_amplitude,
                    .amplitude_saturate = config.slow_harmonic_amplitude_sat,
                    .saturate_max = 1.0f};
  }

  //
  // Generate spine and mush harmonics and noise into it.
  //

}
}  // namespace

SynthesizeResult Synthesize(const SynthesizeConfig& config) {
  float unmask_prob =
      config.unmask_prob_min +
      (util::rndd() * (config.unmask_prob_max - config.unmask_prob_min));

  int full_depth = 2 * config.depth;
  int slice_n = config.height * config.width;

  auto events = CreateArray<bool>(full_depth * slice_n);
  auto mask = CreateArray<bool>(full_depth * slice_n);

retry_break_pattern:
  int offset_3d = 0;
  for (int i = 1; i < full_depth - 1; ++i) {
    if (util::rndd() < config.event_prob) {
      auto break_pattern = BreakPattern(config.width, config.height, 
                                        config.break_pattern_multiplier, 
                                        config.break_pattern_shapes_n,
                                        config.break_pattern_point_plot_prob);
      for (int slice_i = 0; slice_i < slice_n; ++slice_i) {
        events.get()[offset_3d + slice_i] = break_pattern.get()[slice_i];
      }
    } else {
      for (int slice_i = 0; slice_i < slice_n; ++slice_i) {
        events.get()[offset_3d + slice_i] = false;
      }   
    }

    if (util::rndd() < unmask_prob) {
      auto break_pattern = BreakPattern(config.width, config.height,
                                        config.break_pattern_multiplier,
                                        config.break_pattern_shapes_n, 
                                        config.break_pattern_point_plot_prob);
      for (int slice_i = 0; slice_i < slice_n; ++slice_i) {
        mask.get()[offset_3d + slice_i] = break_pattern.get()[slice_i];
      }      
    } else {
      for (int slice_i = 0; slice_i < slice_n; ++slice_i) {
        mask.get()[offset_3d + slice_i] = false;
      }
    }
    offset_3d += slice_n;
  }
  offset_3d = (full_depth - 1) * slice_n;
  for (int slice_i = 0; slice_i < slice_n; ++slice_i) {
    events.get()[slice_i] = true;
    events.get()[offset_3d + slice_i] = true;
  }   

  int min_index_inc = static_cast<int>(config.depth * 0.5);
  int max_index_exc = full_depth - min_index_inc;

  std::vector<std::vector<int>> all_bounds(slice_n, std::vector<int>());
  int event_start_offset = 0;
  for (int y = 0; y < config.height; ++y) {
    for (int x = 0; x < config.width; ++x) {
      int event_offset = event_start_offset + (min_index_inc - 1) * slice_n;
      int first_outside_event = min_index_inc - 1;
      do {
        if (events.get()[event_offset]) break;
        --first_outside_event;
        event_offset -= slice_n;
      } while (first_outside_event >= 0);
       
      event_offset = event_start_offset + max_index_exc * slice_n;
      int last_outside_event = max_index_exc;
      do {
        if (events.get()[event_offset]) break;
        ++last_outside_event;
        event_offset += slice_n;
      } while (last_outside_event < full_depth);

      std::vector<int>* event_times = &(all_bounds[event_start_offset]);
      event_times->push_back(first_outside_event);
      event_offset = event_start_offset + min_index_inc * slice_n;
      for (int i = min_index_inc; i < max_index_exc; ++i) {
        if (events.get()[event_offset]) {
          if ((i - event_times->back()) < config.event_min_period) {
            goto retry_break_pattern;
          }

          event_times->push_back(i);
        }
        event_offset += slice_n;
      }

      if ((last_outside_event - event_times->back()) 
          < config.event_min_period) {
        goto retry_break_pattern;
      }
      event_times->push_back(last_outside_event - min_index_inc);
      for (int i = 0; i < event_times->size() - 1; ++i) {
        (*event_times)[i] -= min_index_inc;
      }

      ++event_start_offset;
    }
  }

  auto mask_trimmed = CreateArray<int8_t>(config.depth * slice_n);
  int dst_offset = 0;
  for (int src_offset = min_index_inc * slice_n;
       src_offset < min_index_inc * slice_n; ++src_offset) {
    mask_trimmed.get()[dst_offset] = mask.get()[src_offset] ? 1 : 0;
  }

  auto raw = CreateArray<float>(config.depth * slice_n);
  auto distance = CreateArray<float>(config.depth * slice_n);
  auto spine = CreateArray<float>(config.depth * slice_n);

  for (int i = 0; i < all_bounds.size(); ++i) {
    GenerateTimeSeries(config, all_bounds[i], raw.get() + i, distance.get() + i, 
                       spine.get() + i, slice_n);
  }

  if (util::TrueWithChance(config.raw_spine_prob)) {
    // Return the spine as the input.
    for (int i = 0; i < (config.depth * slice_n); ++i) {
      mask_trimmed.get()[i] = 1;
      raw.get()[i] = spine.get()[i];
    }    
  } else {
    for (int i = 0; i < (config.depth * slice_n); ++i) {
      raw.get()[i] *= mask_trimmed.get()[i];
    }
  }

  return SynthesizeResult{.raw = std::move(raw),
                          .distance = std::move(distance),
                          .spine = std::move(spine),
                          .mask = std::move(mask_trimmed)};
}

}  // namespace nbd
