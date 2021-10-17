#include "break_pattern.h"

#include <math.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "random.h"

namespace nbd {
namespace {

enum class ShapeFill {
  NONE, FALSE, TRUE
};

inline void Plot(bool value, int x, int y, int dest_width, 
                 std::vector<bool>* dest) {
  (*dest)[y * dest_width + x] = value;
}

void FilledEllipse(int x0, int y0, int x1, int y1, ShapeFill edge,
                   ShapeFill fill, int dest_width, std::vector<bool>* dest) {
  int x_delta = std::abs(x1 - x0);
  int y_delta = std::abs(y1 - y0);
  long x_delta_err = 8 * x_delta * x_delta;
  long y_delta_err = 8 * y_delta * y_delta;

  int y_delta_odd = y_delta & 1; 
  
  long dx = 4 * (1 - x_delta) * y_delta * y_delta;
  long dy = 4 * (y_delta_odd + 1) * x_delta * x_delta; 
  long err = dx + dy + y_delta_odd * x_delta * x_delta;

  if (x0 > x1) std::swap(x0, x1);
  if (y0 > y1) y0 = y1;

  y0 += (y_delta + 1) / 2;
  y1 = y0 - y_delta_odd;

  bool edge_val = edge == ShapeFill::TRUE;
  bool fill_val = fill == ShapeFill::TRUE;
  do {
    if (edge != ShapeFill::NONE) {
      Plot(edge_val, x1, y0, dest_width, dest);
      Plot(edge_val, x0, y0, dest_width, dest);
      Plot(edge_val, x0, y1, dest_width, dest);
      Plot(edge_val, x1, y1, dest_width, dest);
    }
    long err_2 = 2 * err;
    if (err_2 <= dy) {
      if (fill != ShapeFill::NONE) {
        int offset_y0 = y0 * dest_width + x0 + 1;
        int offset_y1 = y1 * dest_width + x0 + 1;
        for (int x_row = x0 + 1; x_row < x1; ++x_row) {
          (*dest)[offset_y0++] = fill_val;
          (*dest)[offset_y1++] = fill_val;
        }
      }

      ++y0;
      --y1;

      dy += x_delta_err;
      err += dy;
    }

    if ((err_2 >= dx) || ((2 * err) > dy)) {
      ++x0;
      --x1;

      dx += y_delta_err;
      err += dx;
    }
  } while (x0 <= x1);

  if (edge == ShapeFill::NONE) return;

  while ((y0 - y1) < y_delta) {
    Plot(edge_val, x0 - 1, y0, dest_width, dest);
    Plot(edge_val, x1 + 1, y0++, dest_width, dest);
    Plot(edge_val, x0 - 1, y1, dest_width, dest);
    Plot(edge_val, x1 + 1, y1--, dest_width, dest);
  }
}

inline ShapeFill RandomShapeFill() {
  static ShapeFill kShapeFillValues[] = {ShapeFill::NONE, ShapeFill::FALSE,
                                         ShapeFill::TRUE};
  return kShapeFillValues[util::rnd() % 3];
}

}  // namespace

std::unique_ptr<bool> BreakPattern(int width, int height, int multiplier,
                                   int shapes_n, float point_plot_prob) {
  int landscape_width = width * multiplier;
  int landscape_height = height * multiplier;
  std::vector<bool> landscape(landscape_height * landscape_width, false);
  for (int shape_i = 0; shape_i < shapes_n; ++shape_i) {
    if (util::TrueWithChance(point_plot_prob)) {
      Plot(util::TrueWithChance(0.5), util::rnd() % landscape_width,
           util::rnd() % landscape_height, landscape_width, &landscape);
    } else {
      FilledEllipse(util::rnd() % landscape_width,
                    util::rnd() % landscape_height,
                    util::rnd() % landscape_width,
                    util::rnd() % landscape_height, RandomShapeFill(), 
                    RandomShapeFill(), landscape_width, &landscape);
    }
  }

  int crop_x = (multiplier != 1) ? util::rnd() % (landscape_width - width) : 0;
  int crop_y =
      (multiplier != 1) ? util::rnd() % (landscape_height - height) : 0;
  
  auto pattern = std::unique_ptr<bool>(new bool[height * width]);
  int src_offset = crop_y * landscape_width + crop_x;
  int src_stride = landscape_width - width;
  int dst_offset = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      pattern.get()[dst_offset++] = landscape[src_offset++];
    }
    src_offset += src_stride;
  }

  return pattern;
}

}  // namespace nbd
