#include "synthesize.h"

#include <memory>

#include "break_pattern.h"
#include "random.h"

namespace nbd {
namespace {

template <typename ElementType>
std::unique_ptr<ElementType> CreateArray(int length) {
  return std::unique_ptr<ElementType>(new ElementType[length]);
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
                                        config.break_pattern_shapes_n);
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
                                        config.break_pattern_shapes_n);
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

  auto mask_trimmed = CreateArray<bool>(config.depth * slice_n);
  int dst_offset = 0;
  for (int src_offset = min_index_inc * slice_n;
       src_offset < min_index_inc * slice_n; ++src_offset) {
    mask_trimmed.get()[dst_offset] = mask.get()[src_offset];
  }
  mask = std::move(mask_trimmed);

  //
  // Create the time-series around all_breaks!
  //

  return SynthesizeResult{.mask = std::move(mask)};
}

}  // namespace nbd
