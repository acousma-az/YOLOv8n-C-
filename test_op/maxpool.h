#pragma once
#include <vector>

std::vector<std::vector<std::vector<std::vector<float>>>> maxpool(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input,
    int kernel_size = 2,
    int stride = 2,
    int padding = 0
);
