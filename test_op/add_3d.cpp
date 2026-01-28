#include "add_3d.h"
#include <stdexcept>
#include <algorithm>

// 三维张量逐元素加法（支持广播）
// input1: [d1, h1, w1]
// input2: [d2, h2, w2]
// 结果: output = input1 + input2
std::vector<std::vector<std::vector<float>>> add_3d(
    const std::vector<std::vector<std::vector<float>>>& input1,
    const std::vector<std::vector<std::vector<float>>>& input2
) {
    // 获取输入1的维度
    int d1 = input1.size();
    int h1 = (d1 > 0) ? input1[0].size() : 0;
    int w1 = (h1 > 0 && d1 > 0) ? input1[0][0].size() : 0;
    
    // 获取输入2的维度
    int d2 = input2.size();
    int h2 = (d2 > 0) ? input2[0].size() : 0;
    int w2 = (h2 > 0 && d2 > 0) ? input2[0][0].size() : 0;
    
    // 检查广播兼容性
    if (!(d1 == d2 || d1 == 1 || d2 == 1)) {
        throw std::invalid_argument("Incompatible dimensions for broadcasting in depth");
    }
    if (!(h1 == h2 || h1 == 1 || h2 == 1)) {
        throw std::invalid_argument("Incompatible dimensions for broadcasting in height");
    }
    if (!(w1 == w2 || w1 == 1 || w2 == 1)) {
        throw std::invalid_argument("Incompatible dimensions for broadcasting in width");
    }
    
    // 计算输出维度
    int d_out = std::max(d1, d2);
    int h_out = std::max(h1, h2);
    int w_out = std::max(w1, w2);
    
    // 初始化输出张量
    std::vector<std::vector<std::vector<float>>> output(
        d_out, 
        std::vector<std::vector<float>>(
            h_out, 
            std::vector<float>(w_out, 0.0f)
        )
    );
    
    // 执行逐元素加法（带广播）
    for (int d = 0; d < d_out; d++) {
        // 计算输入1在深度维度的索引
        int d1_idx = (d1 == 1) ? 0 : (d1 > d) ? d : d % d1;
        // 计算输入2在深度维度的索引
        int d2_idx = (d2 == 1) ? 0 : (d2 > d) ? d : d % d2;
        
        for (int h = 0; h < h_out; h++) {
            // 计算输入1在高度维度的索引
            int h1_idx = (h1 == 1) ? 0 : (h1 > h) ? h : h % h1;
            // 计算输入2在高度维度的索引
            int h2_idx = (h2 == 1) ? 0 : (h2 > h) ? h : h % h2;
            
            for (int w = 0; w < w_out; w++) {
                // 计算输入1在宽度维度的索引
                int w1_idx = (w1 == 1) ? 0 : (w1 > w) ? w : w % w1;
                // 计算输入2在宽度维度的索引
                int w2_idx = (w2 == 1) ? 0 : (w2 > w) ? w : w % w2;
                
                // 执行加法
                output[d][h][w] = input1[d1_idx][h1_idx][w1_idx] + 
                                   input2[d2_idx][h2_idx][w2_idx];
            }
        }
    }
    
    return output;
}
