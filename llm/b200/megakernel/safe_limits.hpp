#pragma once
// Host-side safety limits for the persistent job scheduler.
// Ticket is uint32_t; per-phase completion counters and dependency products are int32_t.
// These bounds do NOT attempt to extend any device counter to 64 bits.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace ofa_limits {
inline int max_safe_steps(int tasks_per_step, const std::vector<int>& phase_tasks,
                          int application_cap = std::numeric_limits<int>::max()) {
  if (tasks_per_step <= 0 || phase_tasks.empty() || application_cap <= 0) return 0;
  std::uint64_t result = std::min<std::uint64_t>(
      std::numeric_limits<std::uint32_t>::max() / static_cast<std::uint64_t>(tasks_per_step),
      static_cast<std::uint64_t>(application_cap));
  for (int n : phase_tasks) {
    if (n < 0) return 0;
    if (n != 0) result = std::min<std::uint64_t>(
        result, std::numeric_limits<std::int32_t>::max() / static_cast<std::uint64_t>(n));
  }
  return static_cast<int>(result);
}
inline bool steps_safe(int steps, int tasks_per_step, const std::vector<int>& phase_tasks) {
  return steps > 0 && steps <= max_safe_steps(tasks_per_step, phase_tasks);
}
}  // namespace ofa_limits
