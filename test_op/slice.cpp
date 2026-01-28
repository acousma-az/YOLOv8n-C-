#include "slice.h"
#include <stdexcept>

// 3D张量slice操作
std::vector<std::vector<std::vector<float>>> slice(
    const std::vector<std::vector<std::vector<float>>>& input,
    int start,
    int end,
    int axis
) {
    if (input.empty()) {
        return {};
    }
    
    // 获取输入维度
    int dim0 = input.size();
    int dim1 = input[0].size();
    int dim2 = input[0][0].size();
    
    // 验证axis参数
    if (axis < 0 || axis >= 3) {
        throw std::invalid_argument("Axis must be 0, 1, or 2 for 3D tensor");
    }
    
    // 根据axis确定slice范围
    if (axis == 0) {
        // 沿第0维slice
        if (start < 0 || end > dim0 || start >= end) {
            throw std::invalid_argument("Invalid slice range for axis 0");
        }
        
        std::vector<std::vector<std::vector<float>>> output(
            end - start,
            std::vector<std::vector<float>>(
                dim1,
                std::vector<float>(dim2)
            )
        );
        
        for (int i = start; i < end; i++) {
            output[i - start] = input[i];
        }
        
        return output;
        
    } else if (axis == 1) {
        // 沿第1维slice
        if (start < 0 || end > dim1 || start >= end) {
            throw std::invalid_argument("Invalid slice range for axis 1");
        }
        
        std::vector<std::vector<std::vector<float>>> output(
            dim0,
            std::vector<std::vector<float>>(
                end - start,
                std::vector<float>(dim2)
            )
        );
        
        for (int d0 = 0; d0 < dim0; d0++) {
            for (int d1 = start; d1 < end; d1++) {
                output[d0][d1 - start] = input[d0][d1];
            }
        }
        
        return output;
        
    } else { // axis == 2
        // 沿第2维slice
        if (start < 0 || end > dim2 || start >= end) {
            throw std::invalid_argument("Invalid slice range for axis 2");
        }
        
        std::vector<std::vector<std::vector<float>>> output(
            dim0,
            std::vector<std::vector<float>>(
                dim1,
                std::vector<float>(end - start)
            )
        );
        
        for (int d0 = 0; d0 < dim0; d0++) {
            for (int d1 = 0; d1 < dim1; d1++) {
                for (int d2 = start; d2 < end; d2++) {
                    output[d0][d1][d2 - start] = input[d0][d1][d2];
                }
            }
        }
        
        return output;
    }
}