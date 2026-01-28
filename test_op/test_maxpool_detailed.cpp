#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "maxpool.h"
#include "npy_loader.h"

// 详细的4D张量数值对比函数
bool detailed_compare_4d(const std::vector<std::vector<std::vector<std::vector<float>>>>& tensor1,
                         const std::vector<std::vector<std::vector<std::vector<float>>>>& tensor2,
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
    
    for (int b = 0; b < tensor1.size(); b++) {
        if (tensor1[b].size() != tensor2[b].size()) {
            std::cout << "❌ 维度1不匹配 at b=" << b << ": " << tensor1[b].size() << " vs " << tensor2[b].size() << std::endl;
            return false;
        }
        
        for (int c = 0; c < tensor1[b].size(); c++) {
            if (tensor1[b][c].size() != tensor2[b][c].size()) {
                std::cout << "❌ 维度2不匹配 at [" << b << "," << c << "]: " 
                         << tensor1[b][c].size() << " vs " << tensor2[b][c].size() << std::endl;
                return false;
            }
            
            for (int h = 0; h < tensor1[b][c].size(); h++) {
                if (tensor1[b][c][h].size() != tensor2[b][c][h].size()) {
                    std::cout << "❌ 维度3不匹配 at [" << b << "," << c << "," << h << "]: " 
                             << tensor1[b][c][h].size() << " vs " << tensor2[b][c][h].size() << std::endl;
                    return false;
                }
                
                for (int w = 0; w < tensor1[b][c][h].size(); w++) {
                    float diff = std::abs(tensor1[b][c][h][w] - tensor2[b][c][h][w]);
                    if (diff > max_diff) max_diff = diff;
                    if (diff > tolerance) {
                        diff_count++;
                        if (diff_count <= 5) { // 只显示前5个差异
                            std::cout << "差异位置 [" << b << "," << c << "," << h << "," << w << "]: "
                                      << std::fixed << std::setprecision(8) 
                                      << tensor1[b][c][h][w] << " vs " << tensor2[b][c][h][w] 
                                      << " (差值: " << diff << ")" << std::endl;
                        }
                    }
                    total_elements++;
                }
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
    int show_count = std::min(5, (int)tensor1[0][0][0].size());
    std::cout << "位置[0,0,0,0-" << (show_count-1) << "]:" << std::endl;
    std::cout << "C++:     ";
    for (int w = 0; w < show_count; w++) {
        std::cout << std::fixed << std::setprecision(6) << tensor1[0][0][0][w] << " ";
    }
    std::cout << std::endl;
    std::cout << "PyTorch: ";
    for (int w = 0; w < show_count; w++) {
        std::cout << std::fixed << std::setprecision(6) << tensor2[0][0][0][w] << " ";
    }
    std::cout << std::endl;
    
    return diff_count == 0;
}

// 测试案例1: 基本2x2 maxpool
void test_maxpool_case1() {
    std::cout << "\n=== 测试案例1: 基本2x2 MaxPool ===" << std::endl;
    
    try {
        auto input = NpyLoader::load_4d_float32("maxpool_case1_input.npy");
        auto expected_result = NpyLoader::load_4d_float32("maxpool_case1_result.npy");
        
        std::cout << "输入形状: [" << input.size() << ", " << input[0].size() << ", " 
                  << input[0][0].size() << ", " << input[0][0][0].size() << "]" << std::endl;
        
        // 执行C++ maxpool
        int kernel_size = 2, stride = 2, padding = 0;
        auto cpp_result = maxpool(input, kernel_size, stride, padding);
        
        std::cout << "PyTorch结果形状: [" << expected_result.size() << ", " << expected_result[0].size() << ", " 
                  << expected_result[0][0].size() << ", " << expected_result[0][0][0].size() << "]" << std::endl;
        std::cout << "C++结果形状: [" << cpp_result.size() << ", " << cpp_result[0].size() << ", " 
                  << cpp_result[0][0].size() << ", " << cpp_result[0][0][0].size() << "]" << std::endl;
        std::cout << "参数: kernel_size=" << kernel_size << ", stride=" << stride << ", padding=" << padding << std::endl;
        
        // 详细对比
        bool match = detailed_compare_4d(cpp_result, expected_result, "基本2x2 MaxPool");
        
        if (match) {
            std::cout << "\n✅ 测试通过: C++MaxPool结果与PyTorch在数值上完全一致!" << std::endl;
        } else {
            std::cout << "\n❌ 测试失败: 发现数值差异!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试失败: " << e.what() << std::endl;
    }
}

// 测试案例2: 3x3 maxpool with padding
void test_maxpool_case2() {
    std::cout << "\n=== 测试案例2: 3x3 MaxPool with padding ===" << std::endl;
    
    try {
        auto input = NpyLoader::load_4d_float32("maxpool_case2_input.npy");
        auto expected_result = NpyLoader::load_4d_float32("maxpool_case2_result.npy");
        
        std::cout << "输入形状: [" << input.size() << ", " << input[0].size() << ", " 
                  << input[0][0].size() << ", " << input[0][0][0].size() << "]" << std::endl;
        
        // 执行C++ maxpool
        int kernel_size = 3, stride = 2, padding = 1;
        auto cpp_result = maxpool(input, kernel_size, stride, padding);
        
        std::cout << "PyTorch结果形状: [" << expected_result.size() << ", " << expected_result[0].size() << ", " 
                  << expected_result[0][0].size() << ", " << expected_result[0][0][0].size() << "]" << std::endl;
        std::cout << "C++结果形状: [" << cpp_result.size() << ", " << cpp_result[0].size() << ", " 
                  << cpp_result[0][0].size() << ", " << cpp_result[0][0][0].size() << "]" << std::endl;
        std::cout << "参数: kernel_size=" << kernel_size << ", stride=" << stride << ", padding=" << padding << std::endl;
        
        // 详细对比
        bool match = detailed_compare_4d(cpp_result, expected_result, "3x3 MaxPool with padding");
        
        if (match) {
            std::cout << "\n✅ 测试通过: C++MaxPool结果与PyTorch在数值上完全一致!" << std::endl;
        } else {
            std::cout << "\n❌ 测试失败: 发现数值差异!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试失败: " << e.what() << std::endl;
    }
}

// 测试案例3: 2x2 maxpool with stride=1
void test_maxpool_case3() {
    std::cout << "\n=== 测试案例3: 2x2 MaxPool with stride=1 ===" << std::endl;
    
    try {
        auto input = NpyLoader::load_4d_float32("maxpool_case3_input.npy");
        auto expected_result = NpyLoader::load_4d_float32("maxpool_case3_result.npy");
        
        std::cout << "输入形状: [" << input.size() << ", " << input[0].size() << ", " 
                  << input[0][0].size() << ", " << input[0][0][0].size() << "]" << std::endl;
        
        // 执行C++ maxpool
        int kernel_size = 2, stride = 1, padding = 0;
        auto cpp_result = maxpool(input, kernel_size, stride, padding);
        
        std::cout << "PyTorch结果形状: [" << expected_result.size() << ", " << expected_result[0].size() << ", " 
                  << expected_result[0][0].size() << ", " << expected_result[0][0][0].size() << "]" << std::endl;
        std::cout << "C++结果形状: [" << cpp_result.size() << ", " << cpp_result[0].size() << ", " 
                  << cpp_result[0][0].size() << ", " << cpp_result[0][0][0].size() << "]" << std::endl;
        std::cout << "参数: kernel_size=" << kernel_size << ", stride=" << stride << ", padding=" << padding << std::endl;
        
        // 详细对比
        bool match = detailed_compare_4d(cpp_result, expected_result, "2x2 MaxPool with stride=1");
        
        if (match) {
            std::cout << "\n✅ 测试通过: C++MaxPool结果与PyTorch在数值上完全一致!" << std::endl;
        } else {
            std::cout << "\n❌ 测试失败: 发现数值差异!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试失败: " << e.what() << std::endl;
    }
}

// 测试案例4: 大尺寸输入
void test_maxpool_case4() {
    std::cout << "\n=== 测试案例4: 大尺寸输入 ===" << std::endl;
    
    try {
        auto input = NpyLoader::load_4d_float32("maxpool_case4_input.npy");
        auto expected_result = NpyLoader::load_4d_float32("maxpool_case4_result.npy");
        
        std::cout << "输入形状: [" << input.size() << ", " << input[0].size() << ", " 
                  << input[0][0].size() << ", " << input[0][0][0].size() << "]" << std::endl;
        
        // 执行C++ maxpool
        int kernel_size = 2, stride = 2, padding = 0;
        auto cpp_result = maxpool(input, kernel_size, stride, padding);
        
        std::cout << "PyTorch结果形状: [" << expected_result.size() << ", " << expected_result[0].size() << ", " 
                  << expected_result[0][0].size() << ", " << expected_result[0][0][0].size() << "]" << std::endl;
        std::cout << "C++结果形状: [" << cpp_result.size() << ", " << cpp_result[0].size() << ", " 
                  << cpp_result[0][0].size() << ", " << cpp_result[0][0][0].size() << "]" << std::endl;
        std::cout << "参数: kernel_size=" << kernel_size << ", stride=" << stride << ", padding=" << padding << std::endl;
        
        // 详细对比
        bool match = detailed_compare_4d(cpp_result, expected_result, "大尺寸输入");
        
        if (match) {
            std::cout << "\n✅ 测试通过: C++MaxPool结果与PyTorch在数值上完全一致!" << std::endl;
        } else {
            std::cout << "\n❌ 测试失败: 发现数值差异!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试失败: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "=== MaxPool 算子详细数值对比测试 ===" << std::endl;
    
    test_maxpool_case1();
    test_maxpool_case2();
    test_maxpool_case3();
    test_maxpool_case4();
    
    std::cout << "\n=== MaxPool测试完成 ===" << std::endl;
    std::cout << "注意: 测试进行了逐元素的数值对比，确保C++MaxPool实现与PyTorch的数学一致性" << std::endl;
    
    return 0;
}
