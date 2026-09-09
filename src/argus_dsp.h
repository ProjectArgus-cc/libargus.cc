#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>

// O(data_size + output_size), O(1) auxiliary memory. Caller owns both extents.
// Validation precedes all writes. Input comprises complete contiguous frames.
inline bool argus_overlap_add(const float * data, int64_t data_size, int64_t output_size,
                              int64_t window, int64_t hop, int64_t padding, float * output) {
    constexpr auto maximum = std::numeric_limits<int64_t>::max();
    if (!data || !output || data_size < 0 || output_size < 0 || window <= 0 || hop <= 0 || padding < 0 ||
        data_size % window != 0 || output_size > maximum / static_cast<int64_t>(sizeof(float)) ||
        data_size > maximum / static_cast<int64_t>(sizeof(float))) return false;
    const int64_t frames = data_size / window;
    if (frames > 0 && frames - 1 > (maximum - window) / hop) return false;
    std::fill_n(output, output_size, 0.0f);
    for (int64_t frame = 0; frame < frames; ++frame) {
        const int64_t start = frame * hop - padding;
        const int64_t begin = std::max<int64_t>(0, start);
        const int64_t end = std::min(output_size, start + window);
        for (int64_t position = begin; position < end; ++position)
            output[position] += data[frame * window + (position - start)];
    }
    return true;
}
