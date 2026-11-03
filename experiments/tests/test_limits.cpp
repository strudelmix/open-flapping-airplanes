#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>
#include "../../llm/b200/megakernel/safe_limits.hpp"

int main() {
  using ofa_limits::max_safe_steps;
  using ofa_limits::steps_safe;
  assert(max_safe_steps(2225, {36, 36, 48, 784, 1}) == 1930322);
  assert(max_safe_steps(2225, {36, 48, 784}, 1 << 20) == (1 << 20));
  assert(max_safe_steps(3601, {36, 48, 784}, 1 << 20) == (1 << 20));
  assert(max_safe_steps(2705, {36, 48, 784}, 1 << 20) == (1 << 20));
  assert(max_safe_steps(1, {1}, 100) == 100);
  assert(max_safe_steps(1, {std::numeric_limits<std::int32_t>::max()}) == 1);
  assert(max_safe_steps(1, {0, 1}) == std::numeric_limits<std::int32_t>::max());
  assert(max_safe_steps(0, {1}) == 0);
  assert(max_safe_steps(100, {}) == 0);
  assert(max_safe_steps(100, {-1}) == 0);
  assert(max_safe_steps(100, {1}, 0) == 0);
  assert(steps_safe(4000, 2225, {784, 36}));
  assert(!steps_safe(1 << 22, 2225, {784, 36}));
  assert(!steps_safe(-1, 2225, {784, 36}));
}
