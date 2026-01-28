#pragma once
#include <vector>

// Concatenate tensors along a given axis
// tensors: vector of 4D tensors, each with shape [batch, channels, height, width]
// axis: concatenation axis (only axis=1 is supported for channel-wise concatenation)
std::vector<std::vector<std::vector<std::vector<float>>>> concat(
    const std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& tensors,
    int axis = 1
);
