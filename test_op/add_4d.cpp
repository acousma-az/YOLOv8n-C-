#include "add_4d.h"
#include <stdexcept>
#include <algorithm>

// 四维张量逐元素加法（支持广播）
// input1: [batch, channels, height, width]
// input2: [batch, channels, height, width] or broadcastable shape
std::vector<std::vector<std::vector<std::vector<float>>>> add_4d(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input1,
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input2
) {
    // 获取输入1的维度
    int b1 = input1.size();
    int c1 = (b1 > 0) ? input1[0].size() : 0;
    int h1 = (c1 > 0 && b1 > 0) ? input1[0][0].size() : 0;
    int w1 = (h1 > 0 && c1 > 0 && b1 > 0) ? input1[0][0][0].size() : 0;
    
    // 获取输入2的维度
    int b2 = input2.size();
    int c2 = (b2 > 0) ? input2[0].size() : 0;
    int h2 = (c2 > 0 && b2 > 0) ? input2[0][0].size() : 0;
    int w2 = (h2 > 0 && c2 > 0 && b2 > 0) ? input2[0][0][0].size() : 0;
    
    // 检查广播兼容性
    if (!(b1 == b2 || b1 == 1 || b2 == 1)) {
        throw std::invalid_argument("Incompatible dimensions for broadcasting in batch");
    }
    if (!(c1 == c2 || c1 == 1 || c2 == 1)) {
        throw std::invalid_argument("Incompatible dimensions for broadcasting in channels");
    }
    if (!(h1 == h2 || h1 == 1 || h2 == 1)) {
        throw std::invalid_argument("Incompatible dimensions for broadcasting in height");
    }
    if (!(w1 == w2 || w1 == 1 || w2 == 1)) {
        throw std::invalid_argument("Incompatible dimensions for broadcasting in width");
    }
    
    // 计算输出维度
    int b_out = std::max(b1, b2);
    int c_out = std::max(c1, c2);
    int h_out = std::max(h1, h2);
    int w_out = std::max(w1, w2);
    
    // 初始化输出张量
    std::vector<std::vector<std::vector<std::vector<float>>>> output(
        b_out, 
        std::vector<std::vector<std::vector<float>>>(
            c_out, 
            std::vector<std::vector<float>>(
                h_out, 
                std::vector<float>(w_out, 0.0f)
            )
        )
    );
    
    // 执行逐元素加法（带广播）
    for (int b = 0; b < b_out; b++) {
        // 计算输入1在batch维度的索引
        int b1_idx = (b1 == 1) ? 0 : (b1 > b) ? b : b % b1;
        // 计算输入2在batch维度的索引
        int b2_idx = (b2 == 1) ? 0 : (b2 > b) ? b : b % b2;
        
        for (int c = 0; c < c_out; c++) {
            // 计算输入1在channels维度的索引
            int c1_idx = (c1 == 1) ? 0 : (c1 > c) ? c : c % c1;
            // 计算输入2在channels维度的索引
            int c2_idx = (c2 == 1) ? 0 : (c2 > c) ? c : c % c2;
            
            for (int h = 0; h < h_out; h++) {
                // 计算输入1在height维度的索引
                int h1_idx = (h1 == 1) ? 0 : (h1 > h) ? h : h % h1;
                // 计算输入2在height维度的索引
                int h2_idx = (h2 == 1) ? 0 : (h2 > h) ? h : h % h2;
                
                for (int w = 0; w < w_out; w++) {
                    // 计算输入1在width维度的索引
                    int w1_idx = (w1 == 1) ? 0 : (w1 > w) ? w : w % w1;
                    // 计算输入2在width维度的索引
                    int w2_idx = (w2 == 1) ? 0 : (w2 > w) ? w : w % w2;
                    
                    // 执行加法
                    output[b][c][h][w] = input1[b1_idx][c1_idx][h1_idx][w1_idx] + 
                                         input2[b2_idx][c2_idx][h2_idx][w2_idx];
                }
            }
        }
    }
    
    return output;
}
