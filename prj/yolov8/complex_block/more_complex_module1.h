#ifndef MORE_COMPLEX_MODULE1_H
#define MORE_COMPLEX_MODULE1_H

#include <vector>

// 复杂数据流模块1：输入[1,128,40,40] -> split + 多路处理 + add操作 + concat -> 输出[1,256,40,40]
// 使用swish_block6系列模块
std::vector<std::vector<std::vector<std::vector<float>>>> more_complex_module1(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input
);

#endif // MORE_COMPLEX_MODULE1_H
