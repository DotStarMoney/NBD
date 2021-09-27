#ifndef NBD_BREAK_PATTERN_H_

#include <memory>
#include <vector>

namespace nbd {

std::unique_ptr<bool> BreakPattern(int width, int height, int multiplier = 6,
                                   int shapes_n = 100);

}  // namespace nbd

#endif  // BREAK_PATTERN_H_
