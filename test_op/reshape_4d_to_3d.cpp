#include <vector>
#include <stdexcept>
#include <numeric>

// 增强版4D张量重塑为3D张量
// input: [batch, channels, height, width]
// target_shape: 目标形状 [dim0, dim1, dim2]，其中一个维度可以是-1（自动计算）
std::vector<std::vector<std::vector<float>>> reshape_4d_to_3d(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input,
    const std::vector<int>& target_shape = {}
) {
    // 验证输入
    if (input.empty()) {
        return {};
    }
    
    // 获取输入维度
    int batch = input.size();
    int channels = (batch > 0) ? input[0].size() : 0;
    int height = (channels > 0 && batch > 0) ? input[0][0].size() : 0;
    int width = (height > 0 && channels > 0 && batch > 0) ? input[0][0][0].size() : 0;
    
    // 检查维度一致性
    for (int b = 0; b < batch; b++) {
        if (input[b].size() != channels) {
            throw std::invalid_argument("Inconsistent channel count across batches");
        }
        
        for (int c = 0; c < channels; c++) {
            if (input[b][c].size() != height) {
                throw std::invalid_argument("Inconsistent height across channels");
            }
            
            for (int h = 0; h < height; h++) {
                if (input[b][c][h].size() != width) {
                    throw std::invalid_argument("Inconsistent width across height");
                }
            }
        }
    }
    
    // 计算总元素数
    int total_elements = batch * channels * height * width;
    
    // 处理默认目标形状 [batch, channels, height*width]
    std::vector<int> final_shape;
    if (target_shape.empty()) {
        final_shape = {batch, channels, height * width};
    } else {
        // 验证目标形状
        if (target_shape.size() != 3) {
            throw std::invalid_argument("Target shape must have exactly 3 dimensions");
        }
        
        // 检查-1的数量
        int minus_one_count = 0;
        for (int dim : target_shape) {
            if (dim == -1) minus_one_count++;
        }
        
        if (minus_one_count > 1) {
            throw std::invalid_argument("Target shape can have at most one -1");
        }
        
        // 计算目标形状的元素数
        int shape_product = 1;
        for (int dim : target_shape) {
            if (dim != -1) {
                if (dim <= 0) {
                    throw std::invalid_argument("All dimensions must be positive or -1");
                }
                shape_product *= dim;
            }
        }
        
        // 处理包含-1的情况
        if (minus_one_count == 1) {
            if (total_elements % shape_product != 0) {
                throw std::invalid_argument("Cannot infer dimension for -1 (not divisible)");
            }
            int inferred_dim = total_elements / shape_product;
            final_shape = target_shape;
            for (int& dim : final_shape) {
                if (dim == -1) dim = inferred_dim;
            }
        } else {
            final_shape = target_shape;
        }
        
        // 验证元素总数匹配
        int final_product = 1;
        for (int dim : final_shape) final_product *= dim;
        
        if (final_product != total_elements) {
            throw std::invalid_argument(
                "Total elements mismatch. Input: " + std::to_string(total_elements) + 
                ", Target: " + std::to_string(final_product)
            );
        }
    }
    
    // 提取目标维度
    int dim0 = final_shape[0];
    int dim1 = final_shape[1];
    int dim2 = final_shape[2];
    
    // 展平输入数据
    std::vector<float> flat_data;
    flat_data.reserve(total_elements);
    
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    flat_data.push_back(input[b][c][h][w]);
                }
            }
        }
    }
    
    // 初始化输出张量
    std::vector<std::vector<std::vector<float>>> output(
        dim0, 
        std::vector<std::vector<float>>(
            dim1, 
            std::vector<float>(dim2, 0.0f)
        )
    );
    
    // 填充输出张量
    int index = 0;
    for (int d0 = 0; d0 < dim0; d0++) {
        for (int d1 = 0; d1 < dim1; d1++) {
            for (int d2 = 0; d2 < dim2; d2++) {
                output[d0][d1][d2] = flat_data[index++];
            }
        }
    }
    
    return output;
}