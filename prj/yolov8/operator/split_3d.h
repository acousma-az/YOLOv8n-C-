#pragma once
#include <vector>

// 沿指定轴分割3D张量
std::vector<std::vector<std::vector<std::vector<float>>>> split_3d(
    const std::vector<std::vector<std::vector<float>>>& input,
    int axis,
    const std::vector<int>& indices
);
