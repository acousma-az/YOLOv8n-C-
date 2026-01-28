#pragma once
#include <vector>

// 3D张量逐元素相减（支持广播）
std::vector<std::vector<std::vector<float>>> sub_3d(
    const std::vector<std::vector<std::vector<float>>>& input1,
    const std::vector<std::vector<std::vector<float>>>& input2
);
