#pragma once
#include <vector>
// 简单数据流模块3：输入[1,128,20,20] -> split + swish处理 + concat -> 输出[1,192,20,20]
// 使用swish_block12_m_0系列模块
std::vector<std::vector<std::vector<std::vector<float>>>> simple_module3(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input
);
