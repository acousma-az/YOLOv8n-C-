#pragma once
#include <vector>

// 3D张量slice操作
std::vector<std::vector<std::vector<float>>> slice(
    const std::vector<std::vector<std::vector<float>>>& input,
    int start,
    int end,
    int axis
);
