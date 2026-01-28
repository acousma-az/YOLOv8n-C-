#include <vector>
#include "../swish_block/swish_block2_m_0_cv1.h"
#include "../swish_block/swish_block2_m_0_cv2.h"
#include "../operator/split.h"
#include "../operator/concat.h"

// 复杂模块函数 - 处理输入张量[batch, 32, 160, 160]
std::vector<std::vector<std::vector<std::vector<float>>>> complex_module0(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input
) {
    // 第一步：将输入分成两份，每份16通道
    std::vector<int> split_sizes = {16, 16}; // 将32通道分为两份，每份16通道
    auto split_outputs = split(input, 1, 0, split_sizes);
    // split_outputs[0]和split_outputs[1]尺寸: [batch, 16, 160, 160]
    
    // 第一份直接进入concat
    auto first_branch = split_outputs[0];  // 尺寸: [batch, 16, 160, 160]
    
    // 第二份有三股路径（每股都是16通道的完整副本）
    // 第一股直接进入concat
    auto second_branch_first = split_outputs[1];  // 尺寸: [batch, 16, 160, 160]
    
    // 第二股直接进入add
    auto second_branch_second = split_outputs[1];  // 尺寸: [batch, 16, 160, 160]

    // 第三股先后经过两个swish block，然后进入add
    auto second_branch_third = split_outputs[1];  // 尺寸: [batch, 16, 160, 160]
    auto swish_output1 = swish_block2_m_0_cv1(second_branch_third);  // 尺寸: [batch, 16, 160, 160]
    auto swish_output2 = swish_block2_m_0_cv2(swish_output1);  // 尺寸: [batch, 16, 160, 160]
    
    // 执行add操作（element-wise addition）
    std::vector<std::vector<std::vector<std::vector<float>>>> add_result;
    add_result.resize(second_branch_second.size());
    for (size_t b = 0; b < second_branch_second.size(); ++b) {
        add_result[b].resize(second_branch_second[b].size());
        for (size_t c = 0; c < second_branch_second[b].size(); ++c) {
            add_result[b][c].resize(second_branch_second[b][c].size());
            for (size_t h = 0; h < second_branch_second[b][c].size(); ++h) {
                add_result[b][c][h].resize(second_branch_second[b][c][h].size());
                for (size_t w = 0; w < second_branch_second[b][c][h].size(); ++w) {
                    add_result[b][c][h][w] = second_branch_second[b][c][h][w] + swish_output2[b][c][h][w];
                }
            }
        }
    }
    
    // 将所有结果连接起来
    std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> to_concat = {
        first_branch,           // 尺寸: [batch, 16, 160, 160]
        second_branch_first,    // 尺寸: [batch, 16, 160, 160] 
        add_result              // 尺寸: [batch, 16, 160, 160]
    };
    auto final_output = concat(to_concat, 1);  // 输出尺寸: [batch, 48, 160, 160]
    
    return final_output;  // 最终输出尺寸: [batch, 48, 160, 160]
}
