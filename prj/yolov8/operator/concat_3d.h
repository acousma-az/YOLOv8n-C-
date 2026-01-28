#pragma once
#include <vector>

// 三维张量连接函数声明
std::vector<std::vector<std::vector<float>>> concat_3d(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& tensors,
    int axis
);
