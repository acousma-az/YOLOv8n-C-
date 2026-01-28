#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "npy_loader.h"
#include "slice.h"

// 辅助函数: 比较两个3D张量
void compare_3d_tensors(
    const std::vector<std::vector<std::vector<float>>>& cpp_result,
    const std::vector<std::vector<std::vector<float>>>& pytorch_result,
    const std::string& test_name,
    float tolerance = 1e-5
) {
    std::cout << "\n=== 详细数值对比分析 (" << test_name << ") ===" << std::endl;
    
    // 检查维度
    if (cpp_result.size() != pytorch_result.size() ||
        (cpp_result.size() > 0 && cpp_result[0].size() != pytorch_result[0].size()) ||
        (cpp_result.size() > 0 && cpp_result[0].size() > 0 && cpp_result[0][0].size() != pytorch_result[0][0].size())) {
        std::cout << "❌ 维度不匹配!" << std::endl;
        return;
    }
    
    int dim0 = cpp_result.size();
    int dim1 = dim0 > 0 ? cpp_result[0].size() : 0;
    int dim2 = dim1 > 0 ? cpp_result[0][0].size() : 0;
    
    float max_diff = 0.0f;
    int total_elements = dim0 * dim1 * dim2;
    int error_count = 0;
    float cpp_sum = 0.0f;
    float pytorch_sum = 0.0f;
    
    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++) {
            for (int k = 0; k < dim2; k++) {
                float cpp_val = cpp_result[i][j][k];
                float pytorch_val = pytorch_result[i][j][k];
                float diff = std::abs(cpp_val - pytorch_val);
                
                max_diff = std::max(max_diff, diff);
                cpp_sum += cpp_val;
                pytorch_sum += pytorch_val;
                
                if (diff > tolerance) {
                    error_count++;
                }
            }
        }
    }
    
    std::cout << "总元素数: " << total_elements << std::endl;
    std::cout << "最大差异: " << std::scientific << max_diff << std::endl;
    std::cout << "超出容差的元素数: " << error_count << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "准确率: " << ((total_elements - error_count) * 100.0f / total_elements) << "%" << std::endl;
    std::cout << std::setprecision(6);
    std::cout << "C++总和: " << cpp_sum << std::endl;
    std::cout << "PyTorch总和: " << pytorch_sum << std::endl;
    
    // 显示部分数值示例
    std::cout << "\n部分数值示例:" << std::endl;
    int show_count = std::min(5, dim2);
    std::cout << "位置[0,0,0-" << (show_count-1) << "]:" << std::endl;
    std::cout << "C++:     ";
    for (int k = 0; k < show_count; k++) {
        std::cout << std::setprecision(6) << std::fixed << cpp_result[0][0][k] << " ";
    }
    std::cout << std::endl;
    std::cout << "PyTorch: ";
    for (int k = 0; k < show_count; k++) {
        std::cout << std::setprecision(6) << std::fixed << pytorch_result[0][0][k] << " ";
    }
    std::cout << std::endl;
    
    if (error_count == 0) {
        std::cout << "\n✅ 测试通过: C++slice结果与PyTorch在数值上完全一致!" << std::endl;
    } else {
        std::cout << "\n❌ 测试失败: 存在超出容差的差异!" << std::endl;
    }
}

int main() {
    std::cout << "=== Slice 算子详细数值对比测试 ===" << std::endl;
    
    try {
        // 测试案例1: 沿axis=0切片
        std::cout << "\n=== 测试案例1: 沿axis=0切片 ===" << std::endl;
        
        auto input1 = load_3d_tensor("slice_case1_input.npy");
        auto pytorch_result1 = load_3d_tensor("slice_case1_result.npy");
        auto params1 = load_params("slice_case1_params.npy");
        
        int start1 = static_cast<int>(params1[0]);
        int end1 = static_cast<int>(params1[1]);
        int axis1 = static_cast<int>(params1[2]);
        
        std::cout << "输入形状: [" << input1.size() << ", " << input1[0].size() << ", " << input1[0][0].size() << "]" << std::endl;
        std::cout << "切片参数: start=" << start1 << ", end=" << end1 << ", axis=" << axis1 << std::endl;
        std::cout << "PyTorch结果形状: [" << pytorch_result1.size() << ", " << pytorch_result1[0].size() << ", " << pytorch_result1[0][0].size() << "]" << std::endl;
        
        auto cpp_result1 = slice(input1, start1, end1, axis1);
        std::cout << "C++结果形状: [" << cpp_result1.size() << ", " << cpp_result1[0].size() << ", " << cpp_result1[0][0].size() << "]" << std::endl;
        
        compare_3d_tensors(cpp_result1, pytorch_result1, "沿axis=0切片");
        
        // 测试案例2: 沿axis=1切片
        std::cout << "\n=== 测试案例2: 沿axis=1切片 ===" << std::endl;
        
        auto input2 = load_3d_tensor("slice_case2_input.npy");
        auto pytorch_result2 = load_3d_tensor("slice_case2_result.npy");
        auto params2 = load_params("slice_case2_params.npy");
        
        int start2 = static_cast<int>(params2[0]);
        int end2 = static_cast<int>(params2[1]);
        int axis2 = static_cast<int>(params2[2]);
        
        std::cout << "输入形状: [" << input2.size() << ", " << input2[0].size() << ", " << input2[0][0].size() << "]" << std::endl;
        std::cout << "切片参数: start=" << start2 << ", end=" << end2 << ", axis=" << axis2 << std::endl;
        
        auto cpp_result2 = slice(input2, start2, end2, axis2);
        
        compare_3d_tensors(cpp_result2, pytorch_result2, "沿axis=1切片");
        
        // 测试案例3: 沿axis=2切片
        std::cout << "\n=== 测试案例3: 沿axis=2切片 ===" << std::endl;
        
        auto input3 = load_3d_tensor("slice_case3_input.npy");
        auto pytorch_result3 = load_3d_tensor("slice_case3_result.npy");
        auto params3 = load_params("slice_case3_params.npy");
        
        int start3 = static_cast<int>(params3[0]);
        int end3 = static_cast<int>(params3[1]);
        int axis3 = static_cast<int>(params3[2]);
        
        std::cout << "输入形状: [" << input3.size() << ", " << input3[0].size() << ", " << input3[0][0].size() << "]" << std::endl;
        std::cout << "切片参数: start=" << start3 << ", end=" << end3 << ", axis=" << axis3 << std::endl;
        
        auto cpp_result3 = slice(input3, start3, end3, axis3);
        
        compare_3d_tensors(cpp_result3, pytorch_result3, "沿axis=2切片");
        
        // 测试案例4: 边界切片(开头)
        std::cout << "\n=== 测试案例4: 边界切片(开头) ===" << std::endl;
        
        auto input4 = load_3d_tensor("slice_case4_input.npy");
        auto pytorch_result4 = load_3d_tensor("slice_case4_result.npy");
        auto params4 = load_params("slice_case4_params.npy");
        
        int start4 = static_cast<int>(params4[0]);
        int end4 = static_cast<int>(params4[1]);
        int axis4 = static_cast<int>(params4[2]);
        
        std::cout << "输入形状: [" << input4.size() << ", " << input4[0].size() << ", " << input4[0][0].size() << "]" << std::endl;
        std::cout << "切片参数: start=" << start4 << ", end=" << end4 << ", axis=" << axis4 << std::endl;
        
        auto cpp_result4 = slice(input4, start4, end4, axis4);
        
        compare_3d_tensors(cpp_result4, pytorch_result4, "边界切片(开头)");
        
        // 测试案例5: 边界切片(结尾)
        std::cout << "\n=== 测试案例5: 边界切片(结尾) ===" << std::endl;
        
        auto input5 = load_3d_tensor("slice_case5_input.npy");
        auto pytorch_result5 = load_3d_tensor("slice_case5_result.npy");
        auto params5 = load_params("slice_case5_params.npy");
        
        int start5 = static_cast<int>(params5[0]);
        int end5 = static_cast<int>(params5[1]);
        int axis5 = static_cast<int>(params5[2]);
        
        std::cout << "输入形状: [" << input5.size() << ", " << input5[0].size() << ", " << input5[0][0].size() << "]" << std::endl;
        std::cout << "切片参数: start=" << start5 << ", end=" << end5 << ", axis=" << axis5 << std::endl;
        
        auto cpp_result5 = slice(input5, start5, end5, axis5);
        
        compare_3d_tensors(cpp_result5, pytorch_result5, "边界切片(结尾)");
        
        std::cout << "\n=== Slice 测试完成 ===" << std::endl;
        std::cout << "注意: 测试进行了逐元素的数值对比，确保C++slice实现与PyTorch的数学一致性" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "测试过程中发生错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
