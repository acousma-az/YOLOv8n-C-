#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "npy_loader.h"
#include "sub_3d.h"

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
        std::cout << "\n✅ 测试通过: C++sub_3d结果与PyTorch在数值上完全一致!" << std::endl;
    } else {
        std::cout << "\n❌ 测试失败: 存在超出容差的差异!" << std::endl;
    }
}

int main() {
    std::cout << "=== Sub_3D 算子详细数值对比测试 ===" << std::endl;
    
    try {
        // 测试案例1: 相同形状的减法
        std::cout << "\n=== 测试案例1: 相同形状的减法 ===" << std::endl;
        
        auto input1_1 = load_3d_tensor("sub3d_case1_input1.npy");
        auto input2_1 = load_3d_tensor("sub3d_case1_input2.npy");
        auto pytorch_result1 = load_3d_tensor("sub3d_case1_result.npy");
        
        std::cout << "输入1形状: [" << input1_1.size() << ", " << input1_1[0].size() << ", " << input1_1[0][0].size() << "]" << std::endl;
        std::cout << "输入2形状: [" << input2_1.size() << ", " << input2_1[0].size() << ", " << input2_1[0][0].size() << "]" << std::endl;
        std::cout << "PyTorch结果形状: [" << pytorch_result1.size() << ", " << pytorch_result1[0].size() << ", " << pytorch_result1[0][0].size() << "]" << std::endl;
        
        auto cpp_result1 = sub_3d(input1_1, input2_1);
        std::cout << "C++结果形状: [" << cpp_result1.size() << ", " << cpp_result1[0].size() << ", " << cpp_result1[0][0].size() << "]" << std::endl;
        
        compare_3d_tensors(cpp_result1, pytorch_result1, "相同形状的减法");
        
        // 测试案例2: 深度维度广播
        std::cout << "\n=== 测试案例2: 深度维度广播 ===" << std::endl;
        
        auto input1_2 = load_3d_tensor("sub3d_case2_input1.npy");
        auto input2_2 = load_3d_tensor("sub3d_case2_input2.npy");
        auto pytorch_result2 = load_3d_tensor("sub3d_case2_result.npy");
        
        std::cout << "输入1形状: [" << input1_2.size() << ", " << input1_2[0].size() << ", " << input1_2[0][0].size() << "]" << std::endl;
        std::cout << "输入2形状: [" << input2_2.size() << ", " << input2_2[0].size() << ", " << input2_2[0][0].size() << "]" << std::endl;
        
        auto cpp_result2 = sub_3d(input1_2, input2_2);
        
        compare_3d_tensors(cpp_result2, pytorch_result2, "深度维度广播");
        
        // 测试案例3: 高度维度广播
        std::cout << "\n=== 测试案例3: 高度维度广播 ===" << std::endl;
        
        auto input1_3 = load_3d_tensor("sub3d_case3_input1.npy");
        auto input2_3 = load_3d_tensor("sub3d_case3_input2.npy");
        auto pytorch_result3 = load_3d_tensor("sub3d_case3_result.npy");
        
        std::cout << "输入1形状: [" << input1_3.size() << ", " << input1_3[0].size() << ", " << input1_3[0][0].size() << "]" << std::endl;
        std::cout << "输入2形状: [" << input2_3.size() << ", " << input2_3[0].size() << ", " << input2_3[0][0].size() << "]" << std::endl;
        
        auto cpp_result3 = sub_3d(input1_3, input2_3);
        
        compare_3d_tensors(cpp_result3, pytorch_result3, "高度维度广播");
        
        // 测试案例4: 宽度维度广播
        std::cout << "\n=== 测试案例4: 宽度维度广播 ===" << std::endl;
        
        auto input1_4 = load_3d_tensor("sub3d_case4_input1.npy");
        auto input2_4 = load_3d_tensor("sub3d_case4_input2.npy");
        auto pytorch_result4 = load_3d_tensor("sub3d_case4_result.npy");
        
        std::cout << "输入1形状: [" << input1_4.size() << ", " << input1_4[0].size() << ", " << input1_4[0][0].size() << "]" << std::endl;
        std::cout << "输入2形状: [" << input2_4.size() << ", " << input2_4[0].size() << ", " << input2_4[0][0].size() << "]" << std::endl;
        
        auto cpp_result4 = sub_3d(input1_4, input2_4);
        
        compare_3d_tensors(cpp_result4, pytorch_result4, "宽度维度广播");
        
        // 测试案例5: 多维度广播
        std::cout << "\n=== 测试案例5: 多维度广播 ===" << std::endl;
        
        auto input1_5 = load_3d_tensor("sub3d_case5_input1.npy");
        auto input2_5 = load_3d_tensor("sub3d_case5_input2.npy");
        auto pytorch_result5 = load_3d_tensor("sub3d_case5_result.npy");
        
        std::cout << "输入1形状: [" << input1_5.size() << ", " << input1_5[0].size() << ", " << input1_5[0][0].size() << "]" << std::endl;
        std::cout << "输入2形状: [" << input2_5.size() << ", " << input2_5[0].size() << ", " << input2_5[0][0].size() << "]" << std::endl;
        
        auto cpp_result5 = sub_3d(input1_5, input2_5);
        
        compare_3d_tensors(cpp_result5, pytorch_result5, "多维度广播");
        
        std::cout << "\n=== Sub_3D 测试完成 ===" << std::endl;
        std::cout << "注意: 测试进行了逐元素的数值对比，确保C++sub_3d实现与PyTorch的数学一致性" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "测试过程中发生错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
