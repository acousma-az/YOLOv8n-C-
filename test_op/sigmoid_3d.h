#pragma once
#include <vector>

// 对3D张量应用sigmoid激活
std::vector<std::vector<std::vector<float>>> sigmoid_3d(
    const std::vector<std::vector<std::vector<float>>>& input
);
