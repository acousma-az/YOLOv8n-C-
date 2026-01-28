#pragma once
#include <vector>

// Split 4D tensor along channel axis (axis=1)
std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> split(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input,
    int axis,
    int num_splits = 0,
    const std::vector<int>& split_sizes = {}
);
