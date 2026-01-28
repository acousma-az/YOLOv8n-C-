#include "split_3d.h"
#include <vector>
#include <stdexcept>
#include <numeric>

// 三维张量分割
// input: 3D张量 [dim0, dim1, dim2]
// axis: 分割轴 (0, 1 或 2)
// num_splits: 均等分割份数 (优先级高于 split_sizes)
// split_sizes: 自定义每份大小的向量
std::vector<std::vector<std::vector<std::vector<float>>>> split_3d(
    const std::vector<std::vector<std::vector<float>>>& input,
    int axis,
    int num_splits,
    const std::vector<int>& split_sizes
) {
    // 验证输入
    if (input.empty()) {
        return {};
    }
    
    // 获取输入维度
    int dim0 = input.size();
    int dim1 = (dim0 > 0) ? input[0].size() : 0;
    int dim2 = (dim1 > 0 && dim0 > 0) ? input[0][0].size() : 0;
    
    // 验证维度一致性
    for (int d0 = 0; d0 < dim0; d0++) {
        if (input[d0].size() != dim1) {
            throw std::invalid_argument("Inconsistent dim1 size across dim0");
        }
        for (int d1 = 0; d1 < dim1; d1++) {
            if (input[d0][d1].size() != dim2) {
                throw std::invalid_argument("Inconsistent dim2 size across dim1");
            }
        }
    }
    
    // 验证轴参数
    if (axis < 0 || axis > 2) {
        throw std::invalid_argument("Axis must be 0, 1, or 2");
    }
    
    // 确定分割方案
    std::vector<int> final_split_sizes;
    int axis_size = (axis == 0) ? dim0 : (axis == 1) ? dim1 : dim2;
    
    if (num_splits > 0) {
        // 均等分割模式
        if (axis_size % num_splits != 0) {
            throw std::invalid_argument("Axis size must be divisible by num_splits");
        }
        int split_size = axis_size / num_splits;
        final_split_sizes = std::vector<int>(num_splits, split_size);
    } else if (!split_sizes.empty()) {
        // 自定义分割模式
        int total = 0;
        for (int size : split_sizes) {
            if (size <= 0) {
                throw std::invalid_argument("All split sizes must be positive");
            }
            total += size;
        }
        if (total != axis_size) {
            throw std::invalid_argument("Sum of split_sizes must equal axis size");
        }
        final_split_sizes = split_sizes;
    } else {
        throw std::invalid_argument("Must specify either num_splits or split_sizes");
    }
    
    // 准备输出向量
    int num_outputs = final_split_sizes.size();
    std::vector<std::vector<std::vector<std::vector<float>>>> outputs;
    outputs.reserve(num_outputs);
    
    // 执行分割操作
    int start_index = 0;
    for (int i = 0; i < num_outputs; i++) {
        int end_index = start_index + final_split_sizes[i];
        
        // 初始化当前输出张量
        std::vector<std::vector<std::vector<float>>> output_tensor;
        
        if (axis == 0) {
            // 沿dim0分割
            output_tensor = std::vector<std::vector<std::vector<float>>>(
                final_split_sizes[i],
                std::vector<std::vector<float>>(
                    dim1,
                    std::vector<float>(dim2, 0.0f)
                )
            );
            
            for (int d0 = start_index, od0 = 0; d0 < end_index; d0++, od0++) {
                output_tensor[od0] = input[d0];
            }
        } 
        else if (axis == 1) {
            // 沿dim1分割
            output_tensor = std::vector<std::vector<std::vector<float>>>(
                dim0,
                std::vector<std::vector<float>>(
                    final_split_sizes[i],
                    std::vector<float>(dim2, 0.0f)
                )
            );
            
            for (int d0 = 0; d0 < dim0; d0++) {
                for (int d1 = start_index, od1 = 0; d1 < end_index; d1++, od1++) {
                    output_tensor[d0][od1] = input[d0][d1];
                }
            }
        } 
        else { // axis == 2
            // 沿dim2分割
            output_tensor = std::vector<std::vector<std::vector<float>>>(
                dim0,
                std::vector<std::vector<float>>(
                    dim1,
                    std::vector<float>(final_split_sizes[i], 0.0f)
                )
            );
            
            for (int d0 = 0; d0 < dim0; d0++) {
                for (int d1 = 0; d1 < dim1; d1++) {
                    for (int d2 = start_index, od2 = 0; d2 < end_index; d2++, od2++) {
                        output_tensor[d0][d1][od2] = input[d0][d1][d2];
                    }
                }
            }
        }
        
        outputs.push_back(std::move(output_tensor));
        start_index = end_index;
    }
    
    return outputs;
}