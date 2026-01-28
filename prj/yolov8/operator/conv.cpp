#include <vector>
#include <iostream>
#include <algorithm>

// Conv2D function
// input: [batch, in_channels, in_height, in_width]
// weight: [out_channels, in_channels, kernel_height, kernel_width]
// bias: [out_channels]
// stride: [stride_h, stride_w]
// padding: [pad_h, pad_w]
std::vector<std::vector<std::vector<std::vector<float>>>> conv(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input,
    const std::vector<std::vector<std::vector<std::vector<float>>>>& weight,
    const std::vector<float>& bias,
    const std::vector<int>& stride,
    const std::vector<int>& padding
) {
    int batch = input.size();
    int in_channels = input[0].size();
    int in_height = input[0][0].size();
    int in_width = input[0][0][0].size();
    
    int out_channels = weight.size();
    int kernel_height = weight[0][0].size();
    int kernel_width = weight[0][0][0].size();
    
    int stride_h = stride[0];
    int stride_w = stride[1];
    int pad_h = padding[0];
    int pad_w = padding[1];
    
    int out_height = (in_height + 2 * pad_h - kernel_height) / stride_h + 1;
    int out_width = (in_width + 2 * pad_w - kernel_width) / stride_w + 1;
    
    // Initialize output tensor
    std::vector<std::vector<std::vector<std::vector<float>>>> output(
        batch, std::vector<std::vector<std::vector<float>>>(
            out_channels, std::vector<std::vector<float>>(
                out_height, std::vector<float>(out_width, 0.0f)
            )
        )
    );
    
    // Convolution operation
    for (int b = 0; b < batch; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    float sum = 0.0f;
                    
                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_height; kh++) {
                            for (int kw = 0; kw < kernel_width; kw++) {
                                int ih = oh * stride_h - pad_h + kh;
                                int iw = ow * stride_w - pad_w + kw;
                                
                                // Check bounds (padding with zeros)
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    sum += input[b][ic][ih][iw] * weight[oc][ic][kh][kw];
                                }
                            }
                        }
                    }
                    
                    // Add bias
                    output[b][oc][oh][ow] = sum + bias[oc];
                }
            }
        }
    }
    
    return output;
}

