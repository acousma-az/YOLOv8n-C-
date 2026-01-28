#pragma once
#include <vector>

// 4D张量重塑为3D张量
std::vector<std::vector<std::vector<float>>> reshape_4d_to_3d(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input,
    const std::vector<int>& target_shape_4d = {}
);
