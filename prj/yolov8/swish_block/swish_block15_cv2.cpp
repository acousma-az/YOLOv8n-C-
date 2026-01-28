#include "swish_block15_cv2.h"
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

std::vector<std::vector<std::vector<std::vector<float>>>> swish_block15_cv2(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input
) {
    // [64, 96, 1, 1]
    std::vector<std::vector<std::vector<std::vector<float>>>> weight(
        64, std::vector<std::vector<std::vector<float>>>(
            96, std::vector<std::vector<float>>(
                1, std::vector<float>(1, 0.0f)
            )
        )
    );
    int idx = 0;
    for (int oc = 0; oc < 64; oc++) {
        for (int ic = 0; ic < 96; ic++) {
            for (int kh = 0; kh < 1; kh++) {
                for (int kw = 0; kw < 1; kw++) {
                    weight[oc][ic][kh][kw] = model_15_cv2_conv_weight[idx++];
                }
            }
        }
    }
    std::vector<float> bias(model_15_cv2_conv_bias, model_15_cv2_conv_bias + 64);
    std::vector<int> stride = {1, 1};
    std::vector<int> padding = {0, 0};
    auto conv_output = conv(input, weight, bias, stride, padding);
    auto sigmoid_output = sigmoid(conv_output);
    auto final_output = mul(conv_output, sigmoid_output);
    return final_output;
}
