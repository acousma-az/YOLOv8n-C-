#pragma once
#include <vector>

// 4D张量softmax
std::vector<std::vector<std::vector<std::vector<float>>>> softmax(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input,
    int axis
);

