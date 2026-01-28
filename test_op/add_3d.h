#ifndef ADD_3D_H
#define ADD_3D_H

#include <vector>

// 三维张量逐元素加法（支持广播）
// input1: [d1, h1, w1]
// input2: [d2, h2, w2]
std::vector<std::vector<std::vector<float>>> add_3d(
    const std::vector<std::vector<std::vector<float>>>& input1,
    const std::vector<std::vector<std::vector<float>>>& input2
);

#endif // ADD_3D_H
