#include <vector>
#include <stdexcept>
#include <numeric>

// 三维张量分割
std::vector<std::vector<std::vector<std::vector<float>>>> split_3d(
    const std::vector<std::vector<std::vector<float>>>& input,
    int axis,
    const std::vector<int>& indices
) {
    // 验证输入
    if (input.empty()) {
        return {};
    }
    
    // 获取输入维度
    size_t dim0 = input.size();
    size_t dim1 = (dim0 > 0) ? input[0].size() : 0;
    size_t dim2 = (dim1 > 0 && dim0 > 0) ? input[0][0].size() : 0;
    
    // 验证维度一致性
    for (size_t d0 = 0; d0 < dim0; d0++) {
        if (input[d0].size() != dim1) {
            throw std::invalid_argument("Inconsistent dim1 size across dim0");
        }
        for (size_t d1 = 0; d1 < dim1; d1++) {
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
    size_t axis_size = (axis == 0) ? dim0 : (axis == 1) ? dim1 : dim2;
    
    // 验证split_sizes
    if (indices.empty()) {
        throw std::invalid_argument("indices cannot be empty");
    }
    
    size_t total = 0;
    for (int size : indices) {
        if (size <= 0) {
            throw std::invalid_argument("All split sizes must be positive");
        }
        total += size;
    }
    if (total != static_cast<size_t>(axis_size)) {
        throw std::invalid_argument("Sum of indices must equal axis size");
    }
    
    // 准备输出向量
    int num_outputs = indices.size();
    std::vector<std::vector<std::vector<std::vector<float>>>> outputs;
    outputs.reserve(num_outputs);
    
    // 执行分割操作
    int start_index = 0;
    for (size_t i = 0; i < static_cast<size_t>(num_outputs); i++) {
        int end_index = start_index + indices[i];
        
        // 初始化当前输出张量
        std::vector<std::vector<std::vector<float>>> output_tensor;
        
        if (axis == 0) {
            // 沿dim0分割
            output_tensor = std::vector<std::vector<std::vector<float>>>(
                indices[i],
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
                    indices[i],
                    std::vector<float>(dim2, 0.0f)
                )
            );
            
            for (size_t d0 = 0; d0 < dim0; d0++) {
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
                    std::vector<float>(indices[i], 0.0f)
                )
            );
            
            for (size_t d0 = 0; d0 < dim0; d0++) {
                for (size_t d1 = 0; d1 < dim1; d1++) {
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