#include "swish_block1.h"
#include "../weight/weights.h"
#include <vector>

// 声明外部函数
extern std::vector<std::vector<std::vector<std::vector<float>>>> conv(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input,
    const std::vector<std::vector<std::vector<std::vector<float>>>>& weight,
    const std::vector<float>& bias,
    const std::vector<int>& stride,
    const std::vector<int>& padding
);

extern std::vector<std::vector<std::vector<std::vector<float>>>> sigmoid(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input
);

extern std::vector<std::vector<std::vector<std::vector<float>>>> mul(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input1,
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input2
);

std::vector<std::vector<std::vector<std::vector<float>>>> swish_block1(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input
) {
    // 将1D权重数组转换为4D张量 [32, 16, 3, 3]
    std::vector<std::vector<std::vector<std::vector<float>>>> weight(
        32, std::vector<std::vector<std::vector<float>>>(
            16, std::vector<std::vector<float>>(
                3, std::vector<float>(3, 0.0f)
            )
        )
    );
    
    // 检查权重数组大小，避免越界访问
    int expected_size = 32 * 16 * 3 * 3; // 4608
    
    int idx = 0;
    for (int oc = 0; oc < 32; oc++) {
        for (int ic = 0; ic < 16; ic++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    if (idx < expected_size) {  // 添加边界检查
                        weight[oc][ic][kh][kw] = model_1_conv_weight[idx++];
                    }
                }
            }
        }
    }

    std::vector<float> bias(model_1_conv_bias, model_1_conv_bias + 32);
    std::vector<int> stride = {2, 2};
    std::vector<int> padding = {1, 1};
    auto conv_output = conv(input, weight, bias, stride, padding);
    auto sigmoid_output = sigmoid(conv_output);
    auto final_output = mul(conv_output, sigmoid_output);
    return final_output;
}
