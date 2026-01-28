#pragma once
#include <vector>

// 对4D张量应用sigmoid激活
std::vector<std::vector<std::vector<std::vector<float>>>> sigmoid(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input
);
