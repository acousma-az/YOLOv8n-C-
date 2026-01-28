#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "concat_3d.h"
#include "npy_loader.h"

// 详细的3D张量数值对比函数
bool detailed_compare_3d(const std::vector<std::vector<std::vector<float>>>& tensor1,
                         const std::vector<std::vector<std::vector<float>>>& tensor2,
                         const std::string& test_name,
                         float tolerance = 1e-5) {
    
    std::cout << "\n=== 详细数值对比分析 (" << test_name << ") ===" << std::endl;
    
    // 维度检查
    if (tensor1.size() != tensor2.size()) {
        std::cout << "❌ 维度0不匹配: " << tensor1.size() << " vs " << tensor2.size() << std::endl;
        return false;
    }
    
    // 计算统计信息
    float max_diff = 0.0f;
    int diff_count = 0;
    int total_elements = 0;
    
    for (int d = 0; d < tensor1.size(); d++) {
        if (tensor1[d].size() != tensor2[d].size()) {
            std::cout << "❌ 维度1不匹配 at d=" << d << ": " << tensor1[d].size() << " vs " << tensor2[d].size() << std::endl;
            return false;
        }
        
        for (int h = 0; h < tensor1[d].size(); h++) {
            if (tensor1[d][h].size() != tensor2[d][h].size()) {
                std::cout << "❌ 维度2不匹配 at [" << d << "," << h << "]: " 
                         << tensor1[d][h].size() << " vs " << tensor2[d][h].size() << std::endl;
                return false;
            }
            
            for (int w = 0; w < tensor1[d][h].size(); w++) {
                float diff = std::abs(tensor1[d][h][w] - tensor2[d][h][w]);
                if (diff > max_diff) max_diff = diff;
                if (diff > tolerance) {
                    diff_count++;
                    if (diff_count <= 5) { // 只显示前5个差异
                        std::cout << "差异位置 [" << d << "," << h << "," << w << "]: "
                                  << std::fixed << std::setprecision(8) 
                                  << tensor1[d][h][w] << " vs " << tensor2[d][h][w] 
                                  << " (差值: " << diff << ")" << std::endl;
                    }
                }
                total_elements++;
            }
        }
    }
    
    // 统计结果
    std::cout << "总元素数: " << total_elements << std::endl;
    std::cout << "最大差异: " << std::scientific << std::setprecision(6) << max_diff << std::endl;
    std::cout << "超出容差的元素数: " << diff_count << std::endl;
    std::cout << "准确率: " << std::fixed << std::setprecision(2) 
              << (100.0 * (total_elements - diff_count) / total_elements) << "%" << std::endl;
    
    // 显示部分数值对比
    std::cout << "\n部分数值示例:" << std::endl;
    int show_count = std::min(5, (int)tensor1[0][0].size());
    std::cout << "位置[0,0,0-" << (show_count-1) << "]:" << std::endl;
    std::cout << "C++:     ";
    for (int w = 0; w < show_count; w++) {
        std::cout << std::fixed << std::setprecision(6) << tensor1[0][0][w] << " ";
    }
    std::cout << std::endl;
    std::cout << "PyTorch: ";
    for (int w = 0; w < show_count; w++) {
        std::cout << std::fixed << std::setprecision(6) << tensor2[0][0][w] << " ";
    }
    std::cout << std::endl;
    
    return diff_count == 0;
}

// 测试案例1: axis=1连接
void test_axis1_concat() {
    std::cout << "\n=== 测试: 沿axis=1连接 ===" << std::endl;
    
    try {
        auto input1 = NpyLoader::load_3d_float32("concat3d_case1_axis1_input1.npy");
        auto input2 = NpyLoader::load_3d_float32("concat3d_case1_axis1_input2.npy");
        auto input3 = NpyLoader::load_3d_float32("concat3d_case1_axis1_input3.npy");
        auto expected_result = NpyLoader::load_3d_float32("concat3d_case1_axis1_result.npy");
        
        std::cout << "输入1 形状: [" << input1.size() << ", " << input1[0].size() << ", " << input1[0][0].size() << "]" << std::endl;
        std::cout << "输入2 形状: [" << input2.size() << ", " << input2[0].size() << ", " << input2[0][0].size() << "]" << std::endl;
        std::cout << "输入3 形状: [" << input3.size() << ", " << input3[0].size() << ", " << input3[0][0].size() << "]" << std::endl;
        
        // 执行C++计算
        std::vector<std::vector<std::vector<std::vector<float>>>> inputs = {input1, input2, input3};
        auto cpp_result = concat_3d(inputs, 1);
        
        std::cout << "PyTorch结果 形状: [" << expected_result.size() << ", " << expected_result[0].size() << ", " << expected_result[0][0].size() << "]" << std::endl;
        std::cout << "C++结果 形状: [" << cpp_result.size() << ", " << cpp_result[0].size() << ", " << cpp_result[0][0].size() << "]" << std::endl;
        
        // 详细对比
        bool match = detailed_compare_3d(cpp_result, expected_result, "axis=1连接");
        
        if (match) {
            std::cout << "\n✅ 测试通过: C++结果与PyTorch在数值上完全一致!" << std::endl;
        } else {
            std::cout << "\n❌ 测试失败: 发现数值差异!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试失败: " << e.what() << std::endl;
    }
}

// 测试案例2: axis=2连接
void test_axis2_concat() {
    std::cout << "\n=== 测试: 沿axis=2连接 ===" << std::endl;
    
    try {
        auto input1 = NpyLoader::load_3d_float32("concat3d_case2_axis2_input1.npy");
        auto input2 = NpyLoader::load_3d_float32("concat3d_case2_axis2_input2.npy");
        auto input3 = NpyLoader::load_3d_float32("concat3d_case2_axis2_input3.npy");
        auto expected_result = NpyLoader::load_3d_float32("concat3d_case2_axis2_result.npy");
        
        std::cout << "输入1 形状: [" << input1.size() << ", " << input1[0].size() << ", " << input1[0][0].size() << "]" << std::endl;
        std::cout << "输入2 形状: [" << input2.size() << ", " << input2[0].size() << ", " << input2[0][0].size() << "]" << std::endl;
        std::cout << "输入3 形状: [" << input3.size() << ", " << input3[0].size() << ", " << input3[0][0].size() << "]" << std::endl;
        
        // 执行C++计算
        std::vector<std::vector<std::vector<std::vector<float>>>> inputs = {input1, input2, input3};
        auto cpp_result = concat_3d(inputs, 2);
        
        std::cout << "PyTorch结果 形状: [" << expected_result.size() << ", " << expected_result[0].size() << ", " << expected_result[0][0].size() << "]" << std::endl;
        std::cout << "C++结果 形状: [" << cpp_result.size() << ", " << cpp_result[0].size() << ", " << cpp_result[0][0].size() << "]" << std::endl;
        
        // 详细对比
        bool match = detailed_compare_3d(cpp_result, expected_result, "axis=2连接");
        
        if (match) {
            std::cout << "\n✅ 测试通过: C++结果与PyTorch在数值上完全一致!" << std::endl;
        } else {
            std::cout << "\n❌ 测试失败: 发现数值差异!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试失败: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "=== Concat_3D 详细数值对比测试 ===" << std::endl;
    
    test_axis1_concat();
    test_axis2_concat();
    
    std::cout << "\n=== 测试完成 ===" << std::endl;
    std::cout << "注意: 这个测试进行了逐元素的数值对比，确保C++实现与PyTorch的数学一致性" << std::endl;
    
    return 0;
}
