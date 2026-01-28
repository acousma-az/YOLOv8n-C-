#pragma once
#include <vector>

// 3D张量重塑为4D张量
std::vector<std::vector<std::vector<std::vector<float>>>> reshape_3d_to_4d(
    const std::vector<std::vector<std::vector<float>>>& input,
    const std::vector<int>& target_shape_4d = {}
);
