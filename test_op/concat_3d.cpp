#include <vector>
#include <stdexcept>
#include <iostream>

// 三维张量连接函数
// tensors: 要连接的三维张量列表
// axis: 连接轴 (1 或 2)
std::vector<std::vector<std::vector<float>>> concat_3d(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& tensors,
    int axis
) {
    // 验证输入
    if (tensors.empty()) {
        return {};
    }
    
    // 验证轴参数
    if (axis != 1 && axis != 2) {
        throw std::invalid_argument("Axis must be 1 or 2 for 3D concatenation");
    }
    
    // 获取参考张量维度
    int dim0_ref = tensors[0].size();
    int dim1_ref = (dim0_ref > 0) ? tensors[0][0].size() : 0;
    int dim2_ref = (dim1_ref > 0 && dim0_ref > 0) ? tensors[0][0][0].size() : 0;
    
    // 检查所有张量非空
    if (dim0_ref == 0 || dim1_ref == 0 || dim2_ref == 0) {
        throw std::invalid_argument("Input tensors cannot be empty");
    }
    
    // 验证非连接维度的一致性
    for (int i = 0; i < tensors.size(); i++) {
        const auto& tensor = tensors[i];
        
        int dim0 = tensor.size();
        int dim1 = (dim0 > 0) ? tensor[0].size() : 0;
        int dim2 = (dim1 > 0 && dim0 > 0) ? tensor[0][0].size() : 0;
        
        if (dim0 == 0 || dim1 == 0 || dim2 == 0) {
            throw std::invalid_argument("Tensor " + std::to_string(i) + " is empty");
        }
        
        // 验证非连接维度匹配
        if (axis == 1) {
            // 连接dim1，要求dim0和dim2相同
            if (dim0 != dim0_ref || dim2 != dim2_ref) {
                throw std::invalid_argument(
                    "Tensor " + std::to_string(i) + " has incompatible dimensions. " +
                    "Expected dim0: " + std::to_string(dim0_ref) + ", got: " + std::to_string(dim0) + ". " +
                    "Expected dim2: " + std::to_string(dim2_ref) + ", got: " + std::to_string(dim2)
                );
            }
        } else { // axis == 2
            // 连接dim2，要求dim0和dim1相同
            if (dim0 != dim0_ref || dim1 != dim1_ref) {
                throw std::invalid_argument(
                    "Tensor " + std::to_string(i) + " has incompatible dimensions. " +
                    "Expected dim0: " + std::to_string(dim0_ref) + ", got: " + std::to_string(dim0) + ". " +
                    "Expected dim1: " + std::to_string(dim1_ref) + ", got: " + std::to_string(dim1)
                );
            }
        }
        
        // 验证内部维度一致性
        for (int d0 = 0; d0 < dim0; d0++) {
            if (tensor[d0].size() != dim1) {
                throw std::invalid_argument(
                    "Tensor " + std::to_string(i) + " has inconsistent dim1 at dim0=" + std::to_string(d0)
                );
            }
            for (int d1 = 0; d1 < dim1; d1++) {
                if (tensor[d0][d1].size() != dim2) {
                    throw std::invalid_argument(
                        "Tensor " + std::to_string(i) + " has inconsistent dim2 at dim0=" + 
                        std::to_string(d0) + ", dim1=" + std::to_string(d1)
                    );
                }
            }
        }
    }
    
    // 计算输出维度
    int out_dim0 = dim0_ref;
    int out_dim1 = dim1_ref;
    int out_dim2 = dim2_ref;
    
    if (axis == 1) {
        out_dim1 = 0;
        for (const auto& tensor : tensors) {
            out_dim1 += tensor[0].size(); // dim1大小
        }
    } else { // axis == 2
        out_dim2 = 0;
        for (const auto& tensor : tensors) {
            out_dim2 += tensor[0][0].size(); // dim2大小
        }
    }
    
    // 初始化输出张量
    std::vector<std::vector<std::vector<float>>> output(
        out_dim0,
        std::vector<std::vector<float>>(
            out_dim1,
            std::vector<float>(out_dim2, 0.0f)
        )
    );
    
    // 执行连接操作
    if (axis == 1) {
        // 沿dim1连接
        for (int d0 = 0; d0 < out_dim0; d0++) {
            int d1_offset = 0;
            for (const auto& tensor : tensors) {
                int tensor_dim1 = tensor[d0].size();
                for (int d1 = 0; d1 < tensor_dim1; d1++) {
                    // 直接复制整行 (dim2维度)
                    output[d0][d1_offset + d1] = tensor[d0][d1];
                }
                d1_offset += tensor_dim1;
            }
        }
    } else { // axis == 2
        // 沿dim2连接
        for (int d0 = 0; d0 < out_dim0; d0++) {
            for (int d1 = 0; d1 < out_dim1; d1++) {
                int d2_offset = 0;
                for (const auto& tensor : tensors) {
                    int tensor_dim2 = tensor[d0][d1].size();
                    for (int d2 = 0; d2 < tensor_dim2; d2++) {
                        output[d0][d1][d2_offset + d2] = tensor[d0][d1][d2];
                    }
                    d2_offset += tensor_dim2;
                }
            }
        }
    }
    
    return output;
}
