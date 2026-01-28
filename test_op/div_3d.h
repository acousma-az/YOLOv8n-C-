#pragma once
#include <vector>

// 三维张量除以常数
std::vector<std::vector<std::vector<float>>> div_3d(
    const std::vector<std::vector<std::vector<float>>>& input,
    float divisor
);
