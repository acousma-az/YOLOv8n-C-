#ifndef SIMPLE_MODULE_H
#define SIMPLE_MODULE_H

#include <vector>

// 简单数据流模块：输入[1,64,80,80] -> split + swish处理 + concat -> 输出[1,96,80,80]
// 使用swish_block15_m_0系列模块
std::vector<std::vector<std::vector<std::vector<float>>>> simple_module0(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input
);

#endif // SIMPLE_MODULE_H
