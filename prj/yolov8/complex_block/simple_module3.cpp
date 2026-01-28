#include <vector>
#include "../swish_block/swish_block12_m_0_cv1.h"
#include "../swish_block/swish_block12_m_0_cv2.h"
#include "../operator/split.h"
#include "../operator/concat.h"
#include "simple_module3.h"

// 简单数据流模块3：输入[1,128,20,20] -> split + swish处理 + concat -> 输出[1,192,20,20]
std::vector<std::vector<std::vector<std::vector<float>>>> simple_module3(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input
) {
    // 第一步：将输入分成两份
    std::vector<int> split_sizes = {64, 64}; // 将128通道分为两份，每份64通道
    auto split_outputs = split(input, 1, 0, split_sizes);
    // split_outputs[0]和split_outputs[1]尺寸: [batch, 64, 20, 20]
    
    // 第一份直接进入concat
    auto first_branch = split_outputs[0];  // 尺寸: [batch, 64, 20, 20]
    
    // 第二份有两股路径
    // 第一股直接进入concat
    auto second_branch_first = split_outputs[1];  // 尺寸: [batch, 64, 20, 20]
    
    // 第二股依次经过两个swish block，然后进入concat
    auto second_branch_second = split_outputs[1];  // 尺寸: [batch, 64, 20, 20]
    auto swish_output1 = swish_block12_m_0_cv1(second_branch_second);  // 尺寸: [batch, 64, 20, 20]
    auto swish_output2 = swish_block12_m_0_cv2(swish_output1);  // 尺寸: [batch, 64, 20, 20]
    
    // 将所有结果连接起来
    std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> to_concat = {
        first_branch,         // 尺寸: [batch, 64, 20, 20]
        second_branch_first,  // 尺寸: [batch, 64, 20, 20]
        swish_output2         // 尺寸: [batch, 64, 20, 20]
    };
    auto final_output = concat(to_concat, 1);  // 输出尺寸: [batch, 192, 20, 20]
    
    return final_output;
}
