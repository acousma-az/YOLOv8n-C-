#include <vector>
#include <stdexcept>
#include <climits>
#include <algorithm>
#include <cfloat> 

// 2D Max Pooling
// input: [batch, channels, height, width]
// kernel_size: 池化窗口大小 (正方形)
// stride: 步长 (默认为kernel_size)
// padding: 填充大小 (默认为0)
std::vector<std::vector<std::vector<std::vector<float>>>> maxpool(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input,
    int kernel_size,
    int stride = 0,
    int padding = 0
) {
    // 验证输入
    if (input.empty() || input[0].empty() || input[0][0].empty() || input[0][0][0].empty()) {
        return {};
    }
    
    // 获取输入维度
    int batch = input.size();
    int channels = input[0].size();
    int height = input[0][0].size();
    int width = input[0][0][0].size();
    

    
    // 如果未指定步长，则使用kernel_size作为步长
    if (stride == 0) {
        stride = kernel_size;
    }
    
    // 计算输出尺寸
    int output_height = (height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (width + 2 * padding - kernel_size) / stride + 1;
    
    // 验证输出尺寸是否有效
    if (output_height <= 0 || output_width <= 0) {
        throw std::invalid_argument("Output dimensions would be non-positive. Adjust parameters.");
    }
    
    // 初始化输出张量
    std::vector<std::vector<std::vector<std::vector<float>>>> output(
        batch, std::vector<std::vector<std::vector<float>>>(
            channels, std::vector<std::vector<float>>(
                output_height, std::vector<float>(output_width, -FLT_MAX)
            )
        )
    );
    
    // 执行池化操作
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < output_height; oh++) {
                for (int ow = 0; ow < output_width; ow++) {
                    // 计算输入窗口起始位置
                    int h_start = oh * stride - padding;
                    int w_start = ow * stride - padding;
                    
                    // 计算输入窗口结束位置
                    int h_end = std::min(h_start + kernel_size, height);
                    int w_end = std::min(w_start + kernel_size, width);
                    
                    // 调整起始位置以处理边界情况
                    h_start = std::max(h_start, 0);
                    w_start = std::max(w_start, 0);
                    
                    // 如果窗口完全在输入之外，跳过
                    if (h_start >= height || w_start >= width || h_end <= h_start || w_end <= w_start) {
                        output[b][c][oh][ow] = 0.0f;
                        continue;
                    }
                    
                    // 在窗口内查找最大值
                    float max_val = -FLT_MAX;
                    for (int ih = h_start; ih < h_end; ih++) {
                        for (int iw = w_start; iw < w_end; iw++) {
                            if (input[b][c][ih][iw] > max_val) {
                                max_val = input[b][c][ih][iw];
                            }
                        }
                    }
                    
                    output[b][c][oh][ow] = max_val;
                }
            }
        }
    }
    
    return output;
}