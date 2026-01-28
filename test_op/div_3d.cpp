#include "div_3d.h"
#include <stdexcept>

// 三维张量除以常数
std::vector<std::vector<std::vector<float>>> div_3d(
    const std::vector<std::vector<std::vector<float>>>& input,
    float divisor
) {
    if (divisor == 0.0f) {
        throw std::invalid_argument("Division by zero");
    }
    
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
    
    // 执行除法
    for (int ch = 0; ch < d; ch++) {
        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                output[ch][row][col] = input[ch][row][col] / divisor;
            }
        }
    }
    
    return output;
}
