#ifndef COMPLEX_MODULE1_H
#define COMPLEX_MODULE1_H

#include <vector>

// 复杂数据流模块：输入[1,256,20,20] -> split + 处理 + add + concat -> 输出[1,384,20,20]
// 使用swish_block8_m_0系列模块
std::vector<std::vector<std::vector<std::vector<float>>>> complex_module1(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input
);

#endif // COMPLEX_MODULE1_H
