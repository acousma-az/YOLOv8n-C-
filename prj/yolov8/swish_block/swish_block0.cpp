#include "swish_block0.h"
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

std::vector<std::vector<std::vector<std::vector<float>>>> swish_block(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input
) {
    // 将1D权重数组转换为4D张量 [16, 3, 3, 3]
    std::vector<std::vector<std::vector<std::vector<float>>>> weight(
        16, std::vector<std::vector<std::vector<float>>>(
            3, std::vector<std::vector<float>>(
                3, std::vector<float>(3, 0.0f)
            )
        )
    );
    
    // 填充权重数据
    int idx = 0;
    for (int oc = 0; oc < 16; oc++) {
        for (int ic = 0; ic < 3; ic++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    weight[oc][ic][kh][kw] = model_0_conv_weight[idx++];
                }
            }
        }
    }
    
    // 将1D bias数组转换为vector
    std::vector<float> bias(model_0_conv_bias, model_0_conv_bias + 16);
    
    // 设置卷积参数：stride=2, padding=1
    std::vector<int> stride = {2, 2};
    std::vector<int> padding = {1, 1};
    
    // 1. 卷积操作
    auto conv_output = conv(input, weight, bias, stride, padding);
    
    // 2. Sigmoid激活
    auto sigmoid_output = sigmoid(conv_output);
    
    // 3. 元素相乘 (Swish = x * sigmoid(x))
    auto final_output = mul(conv_output, sigmoid_output);
    
    return final_output;
}
