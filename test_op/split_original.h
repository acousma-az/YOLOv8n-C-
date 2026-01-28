#ifndef split_H
#define split_H

#include <vector>

// Split tensor along a given axis
// input: [batch, channels, height, width]
// axis: 轴索引 (这里主要处理 axis=1 的通道分割)
// num_splits: 分割份数 (当指定该参数时忽略 split_sizes)
// split_sizes: 每份的大小列表
std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> split(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input,
    int axis,
    int num_splits = 0,
    const std::vector<int>& split_sizes = {}
);

#endif // split_H
