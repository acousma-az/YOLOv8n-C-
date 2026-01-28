#include <vector>
#include "../swish_block/swish_block6_m_0_cv1.h"
#include "../swish_block/swish_block6_m_0_cv2.h"
#include "../swish_block/swish_block6_m_1_cv1.h"
#include "../swish_block/swish_block6_m_1_cv2.h"
#include "../operator/split.h"
#include "../operator/concat.h"
#include "more_complex_module1.h"

// 复杂数据流模块1：输入[1,128,40,40] -> split + 多路处理 + add操作 + concat -> 输出[1,256,40,40]
std::vector<std::vector<std::vector<std::vector<float>>>> more_complex_module1(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input
) {
    // 第一步：将输入分成两份，每份64通道
    std::vector<int> split_sizes = {64, 64}; // 将128通道分为两份，每份64通道
    auto split_outputs = split(input, 1, 0, split_sizes);
    // split_outputs[0]和split_outputs[1]尺寸: [batch, 64, 40, 40]
    
    // 第一份直接进入最终concat
    auto first_branch = split_outputs[0];  // 尺寸: [batch, 64, 40, 40]
    
    // 第二份有三股路径（每股都是64通道的完整副本）
    // 第一股直接进入最终concat
    auto second_branch_first = split_outputs[1];  // 尺寸: [batch, 64, 40, 40]
    
    // 第二股进入add1
    auto second_branch_second = split_outputs[1];  // 尺寸: [batch, 64, 40, 40]

    // 第三股先后经过两个swish block，然后进入add1
    auto second_branch_third = split_outputs[1];  // 尺寸: [batch, 64, 40, 40]
    auto swish_output1_1 = swish_block6_m_0_cv1(second_branch_third);  // 尺寸: [batch, 64, 40, 40]
    auto swish_output1_2 = swish_block6_m_0_cv2(swish_output1_1);  // 尺寸: [batch, 64, 40, 40]
    
    // 执行add1操作（element-wise addition）
    std::vector<std::vector<std::vector<std::vector<float>>>> add1_result;
    add1_result.resize(second_branch_second.size());
    for (size_t b = 0; b < second_branch_second.size(); ++b) {
        add1_result[b].resize(second_branch_second[b].size());
        for (size_t c = 0; c < second_branch_second[b].size(); ++c) {
            add1_result[b][c].resize(second_branch_second[b][c].size());
            for (size_t h = 0; h < second_branch_second[b][c].size(); ++h) {
                add1_result[b][c][h].resize(second_branch_second[b][c][h].size());
                for (size_t w = 0; w < second_branch_second[b][c][h].size(); ++w) {
                    add1_result[b][c][h][w] = second_branch_second[b][c][h][w] + swish_output1_2[b][c][h][w];
                }
            }
        }
    }
    
    // add1的结果也有三股
    // 第一股直接进入最终concat
    auto add1_branch_first = add1_result;  // 尺寸: [batch, 64, 40, 40]
    
    // 第二股进入add2
    auto add1_branch_second = add1_result;  // 尺寸: [batch, 64, 40, 40]
    
    // 第三股先后经过两个swish block，然后进入add2
    auto add1_branch_third = add1_result;  // 尺寸: [batch, 64, 40, 40]
    auto swish_output2_1 = swish_block6_m_1_cv1(add1_branch_third);  // 尺寸: [batch, 64, 40, 40]
    auto swish_output2_2 = swish_block6_m_1_cv2(swish_output2_1);  // 尺寸: [batch, 64, 40, 40]
    
    // 执行add2操作（element-wise addition）
    std::vector<std::vector<std::vector<std::vector<float>>>> add2_result;
    add2_result.resize(add1_branch_second.size());
    for (size_t b = 0; b < add1_branch_second.size(); ++b) {
        add2_result[b].resize(add1_branch_second[b].size());
        for (size_t c = 0; c < add1_branch_second[b].size(); ++c) {
            add2_result[b][c].resize(add1_branch_second[b][c].size());
            for (size_t h = 0; h < add1_branch_second[b][c].size(); ++h) {
                add2_result[b][c][h].resize(add1_branch_second[b][c][h].size());
                for (size_t w = 0; w < add1_branch_second[b][c][h].size(); ++w) {
                    add2_result[b][c][h][w] = add1_branch_second[b][c][h][w] + swish_output2_2[b][c][h][w];
                }
            }
        }
    }
    
    // 将所有结果连接起来
    std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> to_concat = {
        first_branch,         // 尺寸: [batch, 64, 40, 40]
        second_branch_first,  // 尺寸: [batch, 64, 40, 40]
        add1_branch_first,    // 尺寸: [batch, 64, 40, 40]
        add2_result           // 尺寸: [batch, 64, 40, 40]
    };
    auto final_output = concat(to_concat, 1);  // 输出尺寸: [batch, 256, 40, 40]
    
    return final_output;  // 最终输出尺寸: [batch, 256, 40, 40]
}
