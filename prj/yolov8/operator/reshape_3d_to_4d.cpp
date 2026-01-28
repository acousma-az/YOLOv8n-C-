#include <vector>
#include <stdexcept>
#include <numeric>
#include <iostream>

// 3D张量重塑为4D张量
// input: 3D张量 [dim0, dim1, dim2]
// target_shape_4d: 四维目标形状 [batch, channels, height, width]
std::vector<std::vector<std::vector<std::vector<float>>>> reshape_3d_to_4d(
    const std::vector<std::vector<std::vector<float>>>& input,
    const std::vector<int>& target_shape_4d = {}
) {
    // 验证输入
    if (input.empty()) {
        return {};
    }
    
    // 获取输入维度
    size_t dim0 = input.size();
    size_t dim1 = (dim0 > 0) ? input[0].size() : 0;
    size_t dim2 = (dim1 > 0 && dim0 > 0) ? input[0][0].size() : 0;
    
    // 检查维度一致性
    for (size_t d0 = 0; d0 < dim0; d0++) {
        if (input[d0].size() != dim1) {
            throw std::invalid_argument("Inconsistent dim1 across dim0");
        }
        
        for (size_t d1 = 0; d1 < dim1; d1++) {
            if (input[d0][d1].size() != dim2) {
                throw std::invalid_argument("Inconsistent dim2 across dim1");
            }
        }
    }
    
    // 计算总元素数
    size_t total_elements = dim0 * dim1 * dim2;
    
    // 处理目标形状
    std::vector<size_t> final_shape;
    
    if (target_shape_4d.empty()) {
        // 默认行为：[1, dim0, dim1, dim2]
        final_shape = {1, dim0, dim1, dim2};
    } else {
        // 验证四维目标形状
        if (target_shape_4d.size() != 4) {
            throw std::invalid_argument("4D target shape must have exactly 4 dimensions");
        }
        
        // 检查-1的数量
    int minus_one_count = 0;
    for (int dim : target_shape_4d) {
            if (dim == -1) minus_one_count++;
        }
        
        if (minus_one_count > 1) {
            throw std::invalid_argument("4D target shape can have at most one -1");
        }
        
        // 计算目标形状的元素数
    size_t shape_product = 1;
    for (int dim : target_shape_4d) {
            if (dim != -1) {
                if (dim <= 0) {
                    throw std::invalid_argument("All dimensions must be positive or -1");
                }
                shape_product *= dim;
            }
        }
        
        // 处理包含-1的情况
        final_shape = std::vector<size_t>(target_shape_4d.begin(), target_shape_4d.end());
        if (minus_one_count == 1) {
            if (total_elements % shape_product != 0) {
                throw std::invalid_argument("Cannot infer dimension for -1 (not divisible)");
            }
            size_t inferred_dim = total_elements / shape_product;
            for (size_t& dim : final_shape) {
                if (static_cast<long long>(dim) == -1) dim = inferred_dim;
            }
        }
        
        // 验证元素总数匹配
    size_t final_product = 1;
    for (size_t dim : final_shape) final_product *= dim;
        
        if (final_product != total_elements) {
            throw std::invalid_argument(
                "Total elements mismatch. Input: " + std::to_string(total_elements) + 
                ", 4D Target: " + std::to_string(final_product)
            );
        }
    }
    
    // 提取目标维度
    size_t batch = final_shape[0];
    size_t channels = final_shape[1];
    size_t height = final_shape[2];
    size_t width = final_shape[3];
    
    // 展平输入数据
    std::vector<float> flat_data;
    flat_data.reserve(total_elements);
    
    for (size_t d0 = 0; d0 < dim0; d0++) {
        for (size_t d1 = 0; d1 < dim1; d1++) {
            for (size_t d2 = 0; d2 < dim2; d2++) {
                flat_data.push_back(input[d0][d1][d2]);
            }
        }
    }
    
    // 初始化输出张量
    std::vector<std::vector<std::vector<std::vector<float>>>> output(
        batch,
        std::vector<std::vector<std::vector<float>>>(
            channels,
            std::vector<std::vector<float>>(
                height,
                std::vector<float>(width, 0.0f)
            )
        )
    );
    
    // 填充输出张量
    int index = 0;
    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    output[b][c][h][w] = flat_data[index++];
                }
            }
        }
    }
    
    return output;
}