#ifndef SIMPLE_MODULE2_H
#define SIMPLE_MODULE2_H

#include <vector>

// 简单数据流模块2：输入[1,256,20,20] -> split + swish处理 + concat -> 输出[1,384,20,20]
// 使用swish_block21_m_0系列模块
std::vector<std::vector<std::vector<std::vector<float>>>> simple_module2(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input
);

#endif // SIMPLE_MODULE2_H
