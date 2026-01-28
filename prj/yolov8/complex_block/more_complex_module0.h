#ifndef COMPLEX_BLOCK_MODULE1_H
#define COMPLEX_BLOCK_MODULE1_H

#include <vector>

// 复杂数据流模块1：输入[1,64,80,80] -> split + 多路处理 + add操作 + concat
std::vector<std::vector<std::vector<std::vector<float>>>> more_complex_module0(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input
);

#endif // COMPLEX_BLOCK_MODULE1_H
