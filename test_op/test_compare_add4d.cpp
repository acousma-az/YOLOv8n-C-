#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>
#include "add_4d.h"
#include "npy_loader.h"

// 对比4D张量
bool compare_4d_tensors(const std::vector<std::vector<std::vector<std::vector<float>>>>& tensor1,
                        const std::vector<std::vector<std::vector<std::vector<float>>>>& tensor2,
                        float tolerance = 1e-5) {
    if (tensor1.size() != tensor2.size()) return false;
    
    for (int b = 0; b < tensor1.size(); b++) {
        if (tensor1[b].size() != tensor2[b].size()) return false;
        
        for (int c = 0; c < tensor1[b].size(); c++) {
            if (tensor1[b][c].size() != tensor2[b][c].size()) return false;
            
            for (int h = 0; h < tensor1[b][c].size(); h++) {
                if (tensor1[b][c][h].size() != tensor2[b][c][h].size()) return false;
                
                for (int w = 0; w < tensor1[b][c][h].size(); w++) {
                    if (std::abs(tensor1[b][c][h][w] - tensor2[b][c][h][w]) > tolerance) {
                        std::cout << "差异位置 [" << b << "," << c << "," << h << "," << w << "]: "
                                  << tensor1[b][c][h][w] << " vs " << tensor2[b][c][h][w] 
                                  << " (差值: " << std::abs(tensor1[b][c][h][w] - tensor2[b][c][h][w]) << ")" << std::endl;
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

// 打印4D张量信息
void print_4d_tensor_info(const std::vector<std::vector<std::vector<std::vector<float>>>>& tensor, 
                          const std::string& name) {
    std::cout << name << " 形状: [" << tensor.size() << ", " 
              << (tensor.size() > 0 ? tensor[0].size() : 0) << ", " 
              << (tensor.size() > 0 && tensor[0].size() > 0 ? tensor[0][0].size() : 0) << ", "
              << (tensor.size() > 0 && tensor[0].size() > 0 && tensor[0][0].size() > 0 ? tensor[0][0][0].size() : 0) << "]" << std::endl;
}

// 测试单个案例
bool test_case(const std::string& case_name, const std::string& description) {
    std::cout << "=== 测试: " << description << " ===" << std::endl;
    
    try {
        // 加载输入数据
        auto input1 = NpyLoader::load_4d_float32("add4d_" + case_name + "_input1.npy");
        auto input2 = NpyLoader::load_4d_float32("add4d_" + case_name + "_input2.npy");
        auto expected_result = NpyLoader::load_4d_float32("add4d_" + case_name + "_result.npy");
        
        print_4d_tensor_info(input1, "输入1");
        print_4d_tensor_info(input2, "输入2");
        print_4d_tensor_info(expected_result, "PyTorch结果");
        
        // 执行C++计算
        auto cpp_result = add_4d(input1, input2);
        print_4d_tensor_info(cpp_result, "C++结果");
        
        // 对比结果
        bool match = compare_4d_tensors(cpp_result, expected_result);
        
        if (match) {
            std::cout << "✅ 测试通过: C++结果与PyTorch完全一致!" << std::endl;
        } else {
            std::cout << "❌ 测试失败: C++结果与PyTorch不一致!" << std::endl;
        }
        
        std::cout << std::endl;
        return match;
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试失败: " << e.what() << std::endl;
        std::cout << std::endl;
        return false;
    }
}

int main() {
    std::cout << "=== C++ vs PyTorch Add_4D 对比测试 ===" << std::endl << std::endl;
    
    // 测试案例列表
    std::vector<std::pair<std::string, std::string>> test_cases = {
        {"case1_same_shape", "相同形状张量加法"},
        {"case2_batch_broadcast", "Batch维度广播"},
        {"case3_multi_broadcast", "多维度广播"}
    };
    
    int passed = 0;
    int total = test_cases.size();
    
    // 执行所有测试
    for (const auto& test_case_item : test_cases) {
        if (test_case(test_case_item.first, test_case_item.second)) {
            passed++;
        }
    }
    
    // 汇总结果
    std::cout << "=== 测试汇总 ===" << std::endl;
    std::cout << "通过: " << passed << "/" << total << std::endl;
    
    if (passed == total) {
        std::cout << "🎉 所有测试通过! C++实现与PyTorch完全一致!" << std::endl;
        return 0;
    } else {
        std::cout << "⚠️  有测试失败，请检查实现" << std::endl;
        return 1;
    }
}
