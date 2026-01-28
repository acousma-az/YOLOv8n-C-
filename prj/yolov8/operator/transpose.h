#pragma once
#include <vector>

// 4D张量转置
std::vector<std::vector<std::vector<std::vector<float>>>> transpose(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input,
    const std::vector<int>& axes
);
