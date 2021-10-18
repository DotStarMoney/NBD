#include "synthesize.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdint.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <random>

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

float RandomPhase() { return M_PI * 2.0f * util::rndd(); }

constexpr int kSmoothBufferRadius = 7;

void ConvolveSpine(const float* src, float* dest, int dest_stride, int size) {
  // Start accumulation.
  float smooth_acc = 0;
  for (int i = 0; i <= kSmoothBufferRadius; ++i) {
    smooth_acc += src[i];
  }
  float smooth_count = kSmoothBufferRadius + 1;

  // Overlap with first edge.
  for (int i = 0; i < kSmoothBufferRadius; ++i) {
    dest[i * dest_stride] = smooth_acc / smooth_count;
    smooth_acc += src[kSmoothBufferRadius + i + 1];
    smooth_count += 1;
  }

  // Body.
  for (int i = kSmoothBufferRadius; 
       i < (size - kSmoothBufferRadius); 
       ++i) {
    dest[i * dest_stride] = smooth_acc / (kSmoothBufferRadius * 2.f + 1.f);
    smooth_acc += src[i + kSmoothBufferRadius + 1];
    smooth_acc -= src[i - kSmoothBufferRadius];
  }

  // Overlap with last edge.
  for (int i = size - kSmoothBufferRadius + 1; i < size; ++i) {
    smooth_count -= 1;
    smooth_acc -= src[i - kSmoothBufferRadius - 1];
    dest[i * dest_stride] = smooth_acc / smooth_count;
  }
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
                  .phase = RandomPhase(),
                  .amplitude_max = config.base_harmonic_max_amplitude,
                  .amplitude_saturate = config.base_harmonic_amplitude_sat,
                  .saturate_max = config.harmonic_tension_max};
  harmonics[1] = {.frequency = RandomFrequency(
                      config.fast_harmonic_freq_min_days,
                      config.fast_harmonic_freq_max_days), 
                  .phase = RandomPhase(),
                  .amplitude_max = config.fast_harmonic_max_amplitude,
                  .amplitude_saturate = config.fast_harmonic_amplitude_sat,
                  .saturate_max = config.harmonic_tension_max};
  
  if (util::TrueWithChance(config.slow_harmonic_prob)) {
    harmonics_n = 3;
    harmonics[2] = {.frequency = RandomFrequency(
                        config.slow_harmonic_freq_min_years,
                        config.slow_harmonic_freq_max_years), 
                    .phase = RandomPhase(),
                    .amplitude_max = config.slow_harmonic_max_amplitude,
                    .amplitude_saturate = config.slow_harmonic_amplitude_sat,
                    .saturate_max = 1.0f};
  }

  float gaussian_noise_sigma;
  float seg_start_y;
  float seg_end_y;
  float seg_tension_mu;

  int current_seg = 0;
  bool reset_seg_params = true;

  std::default_random_engine slow_generator(
      std::chrono::system_clock::now().time_since_epoch().count());
  std::normal_distribution<float> normal_distribution;

  auto spine_sharp = CreateArray<float>(config.depth);
  for (int i = 0; i < config.depth; ++i) {
    int start_inc = bounds.get()[2 * current_seg];
    int end_ex = bounds.get()[2 * current_seg + 1];

    if (i >= end_ex) {
      current_seg += 1;
      reset_seg_params = true;
    }

    if (reset_seg_params) {
      for (int q = 0; q < harmonics_n; ++q) {
        harmonics[q].GenerateAmplitudeConstants();
      }
      gaussian_noise_sigma =
          config.min_noise_sigma +
          util::rndd() * (config.max_noise_sigma - config.min_noise_sigma);
      seg_start_y = util::rndd();
      seg_end_y = util::rndd();
      seg_tension_mu = RandomTensionMu(config.spine_tension_max);
      reset_seg_params = false;
    }

    spine_sharp.get()[i] =
        seg_start_y +
        (seg_start_y - seg_end_y) *
            powf((i - start_inc) / (end_ex - start_inc), seg_tension_mu);

    raw[i * stride] = 
        gaussian_noise_sigma * normal_distribution(slow_generator);
    for (int q = 0; q < harmonics_n; ++q) {
      raw[i * stride] += harmonics[q].sample(i);
    }
  }

  ConvolveSpine(spine_sharp.get(), spine, stride, config.depth);

  float max_raw = std::numeric_limits<float>::min();
  float min_raw = std::numeric_limits<float>::max();
  if (util::TrueWithChance(config.curve_normalize_prob)) {
    for (int i = 0; i < config.depth; ++i) {
      raw[i * stride] += spine[i * stride];
      if (raw[i * stride] < min_raw) {
        min_raw = raw[i * stride];
      }
      if (raw[i * stride] > max_raw) {
        max_raw = raw[i * stride];
      }
    }
  } else {
    float edge = util::rndd() * config.curve_normalize_rand_edge;
    max_raw = 1.0 + edge;
    min_raw = -edge;
    for (int i = 0; i < config.depth; ++i) {
      raw[i * stride] += spine[i * stride];
    }
  }

  if ((max_raw - min_raw) == 0) {
    max_raw = 1;
    min_raw = 0;
  }

  for (int i = 0; i < config.depth; ++i) {
    raw[i * stride] = (raw[i * stride] - min_raw) / (max_raw - min_raw);
    spine[i * stride] = (spine[i * stride] - min_raw) / (max_raw - min_raw);
  }

  for (int i = 0; i < config.depth; ++i) {
    float x = raw[i * stride];
    if (util::TrueWithChance(config.outlier_prob)) {
      x += (util::rndd() * 2.f - 1.f) * config.outlier_scale;
    }
    if (x < 0) {
      x = 0;
    } else if (x > 1) {
      x = 1;
    }
    raw[i * stride] = x;
  }

  if (events.size() == 2) {
    for (int i = 0; i < config.depth; ++i) {
      distance[i * stride] = 1.0;
    }
  } else {
    int current_mid_break = 0;
    for (int i = 0; i < config.depth; ++i) {
      if ((current_mid_break < (events.size() - 3)) 
          &&
          ((events[current_mid_break + 2] - i) <
           (i - events[current_mid_break + 1]))) {
        current_mid_break++;
      }
    
      float fractional_distance =
          fabsf(events[current_mid_break + 1] - i) / config.max_distance;
      if (fractional_distance > 1.0f) fractional_distance = 1.0f;
      distance[i * stride] = fractional_distance;
    }  
  }
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
       src_offset < max_index_exc * slice_n; ++src_offset) {
    mask_trimmed.get()[dst_offset++] = mask.get()[src_offset] ? 1 : 0;
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
                          .spine = std::move(spine),
                          .distance = std::move(distance),
                          .mask = std::move(mask_trimmed)};
}

}  // namespace nbd
