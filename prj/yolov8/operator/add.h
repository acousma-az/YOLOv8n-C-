#ifndef ADD_H
#define ADD_H

#include <vector>

// Element-wise addition
// input1: [batch, channels, height, width]
// input2: [batch, channels, height, width] or broadcastable shape
std::vector<std::vector<std::vector<std::vector<float>>>> add(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input1,
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input2
);

#endif // ADD_H
