#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cassert>
#include "npy_loader.h"
#include "transpose.h"

void print_tensor_info(const std::vector<std::vector<std::vector<std::vector<float>>>>& tensor, const std::string& name) {
    if (tensor.empty()) {
        std::cout << name << ": 空张量" << std::endl;
        return;
    }
    
    std::cout << name << " 形状: [" << tensor.size() << ", " 
              << tensor[0].size() << ", " 
              << tensor[0][0].size() << ", " 
              << tensor[0][0][0].size() << "]" << std::endl;
    
    // 计算范围
    float min_val = tensor[0][0][0][0];
    float max_val = tensor[0][0][0][0];
    
    for (const auto& batch : tensor) {
        for (const auto& channel : batch) {
            for (const auto& height : channel) {
                for (float val : height) {
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                }
            }
        }
    }
    
    std::cout << name << " 范围: [" << min_val << ", " << max_val << "]" << std::endl;
}

bool compare_tensors(const std::vector<std::vector<std::vector<std::vector<float>>>>& tensor1,
                    const std::vector<std::vector<std::vector<std::vector<float>>>>& tensor2,
                    float tolerance = 1e-5) {
    
    if (tensor1.size() != tensor2.size() ||
        tensor1[0].size() != tensor2[0].size() ||
        tensor1[0][0].size() != tensor2[0][0].size() ||
        tensor1[0][0][0].size() != tensor2[0][0][0].size()) {
        std::cout << "维度不匹配!" << std::endl;
        return false;
    }
    
    float max_diff = 0.0f;
    int diff_count = 0;
    int total_count = 0;
    
    for (size_t b = 0; b < tensor1.size(); ++b) {
        for (size_t c = 0; c < tensor1[0].size(); ++c) {
            for (size_t h = 0; h < tensor1[0][0].size(); ++h) {
                for (size_t w = 0; w < tensor1[0][0][0].size(); ++w) {
                    float diff = std::abs(tensor1[b][c][h][w] - tensor2[b][c][h][w]);
                    max_diff = std::max(max_diff, diff);
                    total_count++;
                    
                    if (diff > tolerance) {
                        diff_count++;
                        if (diff_count <= 10) {  // 只打印前10个差异
                            std::cout << "差异在 [" << b << "," << c << "," << h << "," << w << "]: "
                                     << tensor1[b][c][h][w] << " vs " << tensor2[b][c][h][w] 
                                     << " (diff: " << diff << ")" << std::endl;
                        }
                    }
                }
            }
        }
    }
    
    std::cout << "最大差异: " << max_diff << std::endl;
    std::cout << "超过容忍度的元素: " << diff_count << "/" << total_count << std::endl;
    
    return diff_count == 0;
}

void test_transpose_case(int case_num) {
    std::cout << "\n=== 测试案例 " << case_num << " ===" << std::endl;
    
    // 加载输入数据
    std::string input_file = "transpose_case" + std::to_string(case_num) + "_input.npy";
    std::string result_file = "transpose_case" + std::to_string(case_num) + "_result.npy";
    std::string params_file = "transpose_case" + std::to_string(case_num) + "_params.npy";
    
    auto input_tensor = load_4d_tensor(input_file);
    auto pytorch_result = load_4d_tensor(result_file);
    auto params_data = load_params(params_file);
    
    // 转换参数格式
    std::vector<int> axes;
    for (float val : params_data) {
        axes.push_back(static_cast<int>(val));
    }
    
    std::cout << "输入张量信息:" << std::endl;
    print_tensor_info(input_tensor, "输入");
    
    std::cout << "PyTorch结果信息:" << std::endl;
    print_tensor_info(pytorch_result, "PyTorch结果");
    
    std::cout << "转置参数: [";
    for (size_t i = 0; i < axes.size(); ++i) {
        std::cout << axes[i];
        if (i < axes.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // 执行C++转置
    std::cout << "\n执行C++转置..." << std::endl;
    auto cpp_result = transpose(input_tensor, axes);
    
    std::cout << "C++结果信息:" << std::endl;
    print_tensor_info(cpp_result, "C++结果");
    
    // 比较结果
    std::cout << "\n比较结果:" << std::endl;
    bool match = compare_tensors(cpp_result, pytorch_result);
    
    if (match) {
        std::cout << "✓ 测试案例 " << case_num << " 通过!" << std::endl;
    } else {
        std::cout << "✗ 测试案例 " << case_num << " 失败!" << std::endl;
    }
}

int main() {
    std::cout << "开始 Transpose 算子详细测试" << std::endl;
    std::cout << "========================================" << std::endl;
    
    int passed = 0;
    int total = 5;
    
    // 测试所有案例
    for (int i = 1; i <= total; ++i) {
        try {
            test_transpose_case(i);
            passed++;
        } catch (const std::exception& e) {
            std::cout << "✗ 测试案例 " << i << " 出错: " << e.what() << std::endl;
        }
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "总结: " << passed << "/" << total << " 个测试案例通过" << std::endl;
    
    if (passed == total) {
        std::cout << "🎉 所有 Transpose 测试通过!" << std::endl;
    } else {
        std::cout << "❌ 部分测试失败，需要检查实现" << std::endl;
    }
    
    return 0;
}
