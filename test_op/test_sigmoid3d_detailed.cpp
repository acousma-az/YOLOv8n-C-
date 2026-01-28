#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "sigmoid_3d.h"
#include "npy_loader.h"

// 3D张量数值对比函数
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
    float sum_cpp = 0.0f, sum_pytorch = 0.0f;
    
    for (int d0 = 0; d0 < tensor1.size(); d0++) {
        if (tensor1[d0].size() != tensor2[d0].size()) {
            std::cout << "❌ 维度1不匹配 at d0=" << d0 << ": " << tensor1[d0].size() << " vs " << tensor2[d0].size() << std::endl;
            return false;
        }
        
        for (int d1 = 0; d1 < tensor1[d0].size(); d1++) {
            if (tensor1[d0][d1].size() != tensor2[d0][d1].size()) {
                std::cout << "❌ 维度2不匹配 at [" << d0 << "," << d1 << "]: " 
                         << tensor1[d0][d1].size() << " vs " << tensor2[d0][d1].size() << std::endl;
                return false;
            }
            
            for (int d2 = 0; d2 < tensor1[d0][d1].size(); d2++) {
                float cpp_val = tensor1[d0][d1][d2];
                float pytorch_val = tensor2[d0][d1][d2];
                float diff = std::abs(cpp_val - pytorch_val);
                
                sum_cpp += cpp_val;
                sum_pytorch += pytorch_val;
                
                if (diff > max_diff) max_diff = diff;
                if (diff > tolerance) {
                    diff_count++;
                    if (diff_count <= 5) { // 只显示前5个差异
                        std::cout << "差异位置 [" << d0 << "," << d1 << "," << d2 << "]: "
                                  << std::fixed << std::setprecision(8) 
                                  << cpp_val << " vs " << pytorch_val 
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
    
    // 显示统计信息
    std::cout << "C++总和: " << std::fixed << std::setprecision(6) << sum_cpp << std::endl;
    std::cout << "PyTorch总和: " << std::fixed << std::setprecision(6) << sum_pytorch << std::endl;
    
    // 显示部分数值对比
    std::cout << "\n部分数值示例:" << std::endl;
    int show_count = std::min(5, (int)tensor1[0][0].size());
    std::cout << "位置[0,0,0-" << (show_count-1) << "]:" << std::endl;
    std::cout << "C++:     ";
    for (int d2 = 0; d2 < show_count; d2++) {
        std::cout << std::fixed << std::setprecision(6) << tensor1[0][0][d2] << " ";
    }
    std::cout << std::endl;
    std::cout << "PyTorch: ";
    for (int d2 = 0; d2 < show_count; d2++) {
        std::cout << std::fixed << std::setprecision(6) << tensor2[0][0][d2] << " ";
    }
    std::cout << std::endl;
    
    return diff_count == 0;
}

// 测试案例1: 小尺寸正常值
void test_sigmoid3d_case1() {
    std::cout << "\n=== 测试案例1: 小尺寸正常值 ===" << std::endl;
    
    try {
        auto input = NpyLoader::load_3d_float32("sigmoid3d_case1_input.npy");
        auto expected_result = NpyLoader::load_3d_float32("sigmoid3d_case1_result.npy");
        
        std::cout << "输入形状: [" << input.size() << ", " << input[0].size() 
                  << ", " << input[0][0].size() << "]" << std::endl;
        
        // 显示输入范围
        float min_val = input[0][0][0], max_val = input[0][0][0];
        for (const auto& d0 : input) {
            for (const auto& d1 : d0) {
                for (float val : d1) {
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                }
            }
        }
        std::cout << "输入范围: [" << std::fixed << std::setprecision(6) 
                  << min_val << ", " << max_val << "]" << std::endl;
        
        // 执行C++ sigmoid
        auto cpp_result = sigmoid_3d(input);
        
        std::cout << "PyTorch结果形状: [" << expected_result.size() << ", " << expected_result[0].size() 
                  << ", " << expected_result[0][0].size() << "]" << std::endl;
        std::cout << "C++结果形状: [" << cpp_result.size() << ", " << cpp_result[0].size() 
                  << ", " << cpp_result[0][0].size() << "]" << std::endl;
        
        // 详细对比
        bool match = detailed_compare_3d(cpp_result, expected_result, "小尺寸正常值");
        
        if (match) {
            std::cout << "\n✅ 测试通过: C++sigmoid结果与PyTorch在数值上完全一致!" << std::endl;
        } else {
            std::cout << "\n❌ 测试失败: 发现数值差异!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试失败: " << e.what() << std::endl;
    }
}

// 测试案例2: 极值测试
void test_sigmoid3d_case2() {
    std::cout << "\n=== 测试案例2: 极值测试 ===" << std::endl;
    
    try {
        auto input = NpyLoader::load_3d_float32("sigmoid3d_case2_input.npy");
        auto expected_result = NpyLoader::load_3d_float32("sigmoid3d_case2_result.npy");
        
        std::cout << "输入形状: [" << input.size() << ", " << input[0].size() 
                  << ", " << input[0][0].size() << "]" << std::endl;
        
        // 执行C++ sigmoid
        auto cpp_result = sigmoid_3d(input);
        
        // 详细对比
        bool match = detailed_compare_3d(cpp_result, expected_result, "极值测试");
        
        if (match) {
            std::cout << "\n✅ 测试通过: C++sigmoid结果与PyTorch在数值上完全一致!" << std::endl;
        } else {
            std::cout << "\n❌ 测试失败: 发现数值差异!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试失败: " << e.what() << std::endl;
    }
}

// 测试案例3: 零值和接近零的值
void test_sigmoid3d_case3() {
    std::cout << "\n=== 测试案例3: 零值和接近零的值 ===" << std::endl;
    
    try {
        auto input = NpyLoader::load_3d_float32("sigmoid3d_case3_input.npy");
        auto expected_result = NpyLoader::load_3d_float32("sigmoid3d_case3_result.npy");
        
        std::cout << "输入形状: [" << input.size() << ", " << input[0].size() 
                  << ", " << input[0][0].size() << "]" << std::endl;
        
        // 执行C++ sigmoid
        auto cpp_result = sigmoid_3d(input);
        
        // 详细对比
        bool match = detailed_compare_3d(cpp_result, expected_result, "零值和接近零的值");
        
        if (match) {
            std::cout << "\n✅ 测试通过: C++sigmoid结果与PyTorch在数值上完全一致!" << std::endl;
        } else {
            std::cout << "\n❌ 测试失败: 发现数值差异!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试失败: " << e.what() << std::endl;
    }
}

// 测试案例4: 大尺寸随机值
void test_sigmoid3d_case4() {
    std::cout << "\n=== 测试案例4: 大尺寸随机值 ===" << std::endl;
    
    try {
        auto input = NpyLoader::load_3d_float32("sigmoid3d_case4_input.npy");
        auto expected_result = NpyLoader::load_3d_float32("sigmoid3d_case4_result.npy");
        
        std::cout << "输入形状: [" << input.size() << ", " << input[0].size() 
                  << ", " << input[0][0].size() << "]" << std::endl;
        
        // 执行C++ sigmoid
        auto cpp_result = sigmoid_3d(input);
        
        // 详细对比
        bool match = detailed_compare_3d(cpp_result, expected_result, "大尺寸随机值");
        
        if (match) {
            std::cout << "\n✅ 测试通过: C++sigmoid结果与PyTorch在数值上完全一致!" << std::endl;
        } else {
            std::cout << "\n❌ 测试失败: 发现数值差异!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试失败: " << e.what() << std::endl;
    }
}

// 测试案例5: 特殊数值稳定性测试
void test_sigmoid3d_case5() {
    std::cout << "\n=== 测试案例5: 特殊数值稳定性测试 ===" << std::endl;
    
    try {
        auto input = NpyLoader::load_3d_float32("sigmoid3d_case5_input.npy");
        auto expected_result = NpyLoader::load_3d_float32("sigmoid3d_case5_result.npy");
        
        std::cout << "输入形状: [" << input.size() << ", " << input[0].size() 
                  << ", " << input[0][0].size() << "]" << std::endl;
        
        // 执行C++ sigmoid
        auto cpp_result = sigmoid_3d(input);
        
        // 详细对比
        bool match = detailed_compare_3d(cpp_result, expected_result, "特殊数值稳定性测试");
        
        if (match) {
            std::cout << "\n✅ 测试通过: C++sigmoid结果与PyTorch在数值上完全一致!" << std::endl;
        } else {
            std::cout << "\n❌ 测试失败: 发现数值差异!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试失败: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "=== Sigmoid 3D 算子详细数值对比测试 ===" << std::endl;
    
    test_sigmoid3d_case1();
    test_sigmoid3d_case2();
    test_sigmoid3d_case3();
    test_sigmoid3d_case4();
    test_sigmoid3d_case5();
    
    std::cout << "\n=== Sigmoid 3D 测试完成 ===" << std::endl;
    std::cout << "注意: 测试进行了逐元素的数值对比，确保C++sigmoid实现与PyTorch的数学一致性" << std::endl;
    
    return 0;
}
