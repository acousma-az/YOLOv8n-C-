#ifndef ADD_4D_H
#define ADD_4D_H

#include <vector>

// 四维张量逐元素加法（支持广播）
// input1: [batch, channels, height, width]
// input2: [batch, channels, height, width] or broadcastable shape
std::vector<std::vector<std::vector<std::vector<float>>>> add_4d(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input1,
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input2
);

#endif // ADD_4D_H
