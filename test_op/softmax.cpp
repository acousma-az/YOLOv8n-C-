#include "softmax.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cfloat>

// 4D张量softmax
std::vector<std::vector<std::vector<std::vector<float>>>> softmax(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input,
    int axis
) {
    if (input.empty()) {
        return {};
    }
    
    // 验证axis参数
    if (axis < 0 || axis >= 4) {
        throw std::invalid_argument("Axis must be in range [0, 3] for 4D tensor");
    }
    
    // 获取输入维度
    int dim0 = input.size();
    int dim1 = input[0].size();
    int dim2 = input[0][0].size();
    int dim3 = input[0][0][0].size();
    
    // 初始化输出张量
    std::vector<std::vector<std::vector<std::vector<float>>>> output(
        dim0,
        std::vector<std::vector<std::vector<float>>>(
            dim1,
            std::vector<std::vector<float>>(
                dim2,
                std::vector<float>(dim3, 0.0f)
            )
        )
    );
    
    // 根据axis执行softmax
    if (axis == 0) {
        // 沿dim0方向softmax
        for (int d1 = 0; d1 < dim1; d1++) {
            for (int d2 = 0; d2 < dim2; d2++) {
                for (int d3 = 0; d3 < dim3; d3++) {
                    // 找到最大值用于数值稳定性
                    float max_val = -FLT_MAX;
                    for (int d0 = 0; d0 < dim0; d0++) {
                        max_val = std::max(max_val, input[d0][d1][d2][d3]);
                    }
                    
                    // 计算exp值和sum
                    float sum = 0.0f;
                    for (int d0 = 0; d0 < dim0; d0++) {
                        output[d0][d1][d2][d3] = std::exp(input[d0][d1][d2][d3] - max_val);
                        sum += output[d0][d1][d2][d3];
                    }
                    
                    // 归一化
                    for (int d0 = 0; d0 < dim0; d0++) {
                        output[d0][d1][d2][d3] /= sum;
                    }
                }
            }
        }
    } else if (axis == 1) {
        // 沿dim1方向softmax
        for (int d0 = 0; d0 < dim0; d0++) {
            for (int d2 = 0; d2 < dim2; d2++) {
                for (int d3 = 0; d3 < dim3; d3++) {
                    // 找到最大值
                    float max_val = -FLT_MAX;
                    for (int d1 = 0; d1 < dim1; d1++) {
                        max_val = std::max(max_val, input[d0][d1][d2][d3]);
                    }
                    
                    // 计算exp值和sum
                    float sum = 0.0f;
                    for (int d1 = 0; d1 < dim1; d1++) {
                        output[d0][d1][d2][d3] = std::exp(input[d0][d1][d2][d3] - max_val);
                        sum += output[d0][d1][d2][d3];
                    }
                    
                    // 归一化
                    for (int d1 = 0; d1 < dim1; d1++) {
                        output[d0][d1][d2][d3] /= sum;
                    }
                }
            }
        }
    } else if (axis == 2) {
        // 沿dim2方向softmax
        for (int d0 = 0; d0 < dim0; d0++) {
            for (int d1 = 0; d1 < dim1; d1++) {
                for (int d3 = 0; d3 < dim3; d3++) {
                    // 找到最大值
                    float max_val = -FLT_MAX;
                    for (int d2 = 0; d2 < dim2; d2++) {
                        max_val = std::max(max_val, input[d0][d1][d2][d3]);
                    }
                    
                    // 计算exp值和sum
                    float sum = 0.0f;
                    for (int d2 = 0; d2 < dim2; d2++) {
                        output[d0][d1][d2][d3] = std::exp(input[d0][d1][d2][d3] - max_val);
                        sum += output[d0][d1][d2][d3];
                    }
                    
                    // 归一化
                    for (int d2 = 0; d2 < dim2; d2++) {
                        output[d0][d1][d2][d3] /= sum;
                    }
                }
            }
        }
    } else { // axis == 3
        // 沿dim3方向softmax
        for (int d0 = 0; d0 < dim0; d0++) {
            for (int d1 = 0; d1 < dim1; d1++) {
                for (int d2 = 0; d2 < dim2; d2++) {
                    // 找到最大值
                    float max_val = -FLT_MAX;
                    for (int d3 = 0; d3 < dim3; d3++) {
                        max_val = std::max(max_val, input[d0][d1][d2][d3]);
                    }
                    
                    // 计算exp值和sum
                    float sum = 0.0f;
                    for (int d3 = 0; d3 < dim3; d3++) {
                        output[d0][d1][d2][d3] = std::exp(input[d0][d1][d2][d3] - max_val);
                        sum += output[d0][d1][d2][d3];
                    }
                    
                    // 归一化
                    for (int d3 = 0; d3 < dim3; d3++) {
                        output[d0][d1][d2][d3] /= sum;
                    }
                }
            }
        }
    }
    
    return output;
}
