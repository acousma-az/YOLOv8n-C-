#ifndef SIMPLE_MODULE1_H
#define SIMPLE_MODULE1_H

#include <vector>

// 简单数据流模块1：输入[1,128,80,80] -> split + swish处理 + concat -> 输出[1,192,80,80]
// 使用swish_block18_m_0系列模块
std::vector<std::vector<std::vector<std::vector<float>>>> simple_module1(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input
);

#endif // SIMPLE_MODULE1_H
