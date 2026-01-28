#pragma once
#include <vector>

// 沿指定轴分割3D张量
// 支持均等分割和自定义分割
std::vector<std::vector<std::vector<std::vector<float>>>> split_3d(
    const std::vector<std::vector<std::vector<float>>>& input,
    int axis,
    int num_splits = 0,
    const std::vector<int>& split_sizes = {}
);
