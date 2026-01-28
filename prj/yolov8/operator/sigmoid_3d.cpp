#include "sigmoid_3d.h"
#include <cmath>

// 对3D张量应用sigmoid激活
std::vector<std::vector<std::vector<float>>> sigmoid_3d(
    const std::vector<std::vector<std::vector<float>>>& input
) {
    // 获取输入维度
    int d = input.size();
    int h = (d > 0) ? input[0].size() : 0;
    int w = (h > 0 && d > 0) ? input[0][0].size() : 0;
    
    // 初始化输出张量
    std::vector<std::vector<std::vector<float>>> output(
        d, 
        std::vector<std::vector<float>>(
            h, 
            std::vector<float>(w, 0.0f)
        )
    );
    
    // 应用sigmoid函数: sigmoid(x) = 1 / (1 + exp(-x))
    for (int d_idx = 0; d_idx < d; d_idx++) {
        for (int h_idx = 0; h_idx < h; h_idx++) {
            for (int w_idx = 0; w_idx < w; w_idx++) {
                float x = input[d_idx][h_idx][w_idx];
                output[d_idx][h_idx][w_idx] = 1.0f / (1.0f + std::exp(-x));
            }
        }
    }
    
    return output;
}