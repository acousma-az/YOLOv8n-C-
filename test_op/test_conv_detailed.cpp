#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "conv.h"
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

// 加载1D向量（偏置）
std::vector<float> load_1d_vector(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // 跳过numpy头部
    NpyLoader::skip_npy_header(file);
    
    // 根据文件名确定大小
    int size;
    if (filename.find("conv_case1_bias") != std::string::npos) {
        size = 16;
    } else if (filename.find("conv_case2_bias") != std::string::npos) {
        size = 64;
    } else if (filename.find("conv_case3_bias") != std::string::npos) {
        size = 16;
    } else {
        throw std::runtime_error("Unknown bias file pattern: " + filename);
    }
    
    std::vector<float> vec(size);
    for (int i = 0; i < size; i++) {
        file.read(reinterpret_cast<char*>(&vec[i]), sizeof(float));
        if (file.fail()) {
            throw std::runtime_error("Failed to read bias data from file: " + filename);
        }
    }
    
    std::cout << "成功加载偏置 " << filename << " 大小: " << size << std::endl;
    return vec;
}

// 测试案例1: 基本3x3卷积
void test_conv_case1() {
    std::cout << "\n=== 测试案例1: 基本3x3卷积 ===" << std::endl;
    
    try {
        auto input = NpyLoader::load_4d_float32("conv_case1_input.npy");
        auto weight = NpyLoader::load_4d_float32("conv_case1_weight.npy");
        auto bias = load_1d_vector("conv_case1_bias.npy");
        auto expected_result = NpyLoader::load_4d_float32("conv_case1_result.npy");
        
        std::cout << "输入形状: [" << input.size() << ", " << input[0].size() << ", " 
                  << input[0][0].size() << ", " << input[0][0][0].size() << "]" << std::endl;
        std::cout << "权重形状: [" << weight.size() << ", " << weight[0].size() << ", " 
                  << weight[0][0].size() << ", " << weight[0][0][0].size() << "]" << std::endl;
        std::cout << "偏置大小: " << bias.size() << std::endl;
        
        // 执行C++卷积
        std::vector<int> stride = {1, 1};
        std::vector<int> padding = {1, 1};
        auto cpp_result = conv(input, weight, bias, stride, padding);
        
        std::cout << "PyTorch结果形状: [" << expected_result.size() << ", " << expected_result[0].size() << ", " 
                  << expected_result[0][0].size() << ", " << expected_result[0][0][0].size() << "]" << std::endl;
        std::cout << "C++结果形状: [" << cpp_result.size() << ", " << cpp_result[0].size() << ", " 
                  << cpp_result[0][0].size() << ", " << cpp_result[0][0][0].size() << "]" << std::endl;
        
        // 详细对比
        bool match = detailed_compare_4d(cpp_result, expected_result, "基本3x3卷积");
        
        if (match) {
            std::cout << "\n✅ 测试通过: C++卷积结果与PyTorch在数值上完全一致!" << std::endl;
        } else {
            std::cout << "\n❌ 测试失败: 发现数值差异!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试失败: " << e.what() << std::endl;
    }
}

// 测试案例2: 1x1卷积
void test_conv_case2() {
    std::cout << "\n=== 测试案例2: 1x1卷积 ===" << std::endl;
    
    try {
        auto input = NpyLoader::load_4d_float32("conv_case2_input.npy");
        auto weight = NpyLoader::load_4d_float32("conv_case2_weight.npy");
        auto bias = load_1d_vector("conv_case2_bias.npy");
        auto expected_result = NpyLoader::load_4d_float32("conv_case2_result.npy");
        
        std::cout << "输入形状: [" << input.size() << ", " << input[0].size() << ", " 
                  << input[0][0].size() << ", " << input[0][0][0].size() << "]" << std::endl;
        std::cout << "权重形状: [" << weight.size() << ", " << weight[0].size() << ", " 
                  << weight[0][0].size() << ", " << weight[0][0][0].size() << "]" << std::endl;
        std::cout << "偏置大小: " << bias.size() << std::endl;
        
        // 执行C++卷积
        std::vector<int> stride = {1, 1};
        std::vector<int> padding = {0, 0};
        auto cpp_result = conv(input, weight, bias, stride, padding);
        
        std::cout << "PyTorch结果形状: [" << expected_result.size() << ", " << expected_result[0].size() << ", " 
                  << expected_result[0][0].size() << ", " << expected_result[0][0][0].size() << "]" << std::endl;
        std::cout << "C++结果形状: [" << cpp_result.size() << ", " << cpp_result[0].size() << ", " 
                  << cpp_result[0][0].size() << ", " << cpp_result[0][0][0].size() << "]" << std::endl;
        
        // 详细对比
        bool match = detailed_compare_4d(cpp_result, expected_result, "1x1卷积");
        
        if (match) {
            std::cout << "\n✅ 测试通过: C++卷积结果与PyTorch在数值上完全一致!" << std::endl;
        } else {
            std::cout << "\n❌ 测试失败: 发现数值差异!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试失败: " << e.what() << std::endl;
    }
}

// 测试案例3: stride=2卷积
void test_conv_case3() {
    std::cout << "\n=== 测试案例3: stride=2卷积 ===" << std::endl;
    
    try {
        auto input = NpyLoader::load_4d_float32("conv_case3_input.npy");
        auto weight = NpyLoader::load_4d_float32("conv_case3_weight.npy");
        auto bias = load_1d_vector("conv_case3_bias.npy");
        auto expected_result = NpyLoader::load_4d_float32("conv_case3_result.npy");
        
        std::cout << "输入形状: [" << input.size() << ", " << input[0].size() << ", " 
                  << input[0][0].size() << ", " << input[0][0][0].size() << "]" << std::endl;
        std::cout << "权重形状: [" << weight.size() << ", " << weight[0].size() << ", " 
                  << weight[0][0].size() << ", " << weight[0][0][0].size() << "]" << std::endl;
        std::cout << "偏置大小: " << bias.size() << std::endl;
        
        // 执行C++卷积
        std::vector<int> stride = {2, 2};
        std::vector<int> padding = {1, 1};
        auto cpp_result = conv(input, weight, bias, stride, padding);
        
        std::cout << "PyTorch结果形状: [" << expected_result.size() << ", " << expected_result[0].size() << ", " 
                  << expected_result[0][0].size() << ", " << expected_result[0][0][0].size() << "]" << std::endl;
        std::cout << "C++结果形状: [" << cpp_result.size() << ", " << cpp_result[0].size() << ", " 
                  << cpp_result[0][0].size() << ", " << cpp_result[0][0][0].size() << "]" << std::endl;
        
        // 详细对比
        bool match = detailed_compare_4d(cpp_result, expected_result, "stride=2卷积");
        
        if (match) {
            std::cout << "\n✅ 测试通过: C++卷积结果与PyTorch在数值上完全一致!" << std::endl;
        } else {
            std::cout << "\n❌ 测试失败: 发现数值差异!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试失败: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "=== Conv 算子详细数值对比测试 ===" << std::endl;
    
    test_conv_case1();
    test_conv_case2();
    test_conv_case3();
    
    std::cout << "\n=== Conv测试完成 ===" << std::endl;
    std::cout << "注意: 测试进行了逐元素的数值对比，确保C++卷积实现与PyTorch的数学一致性" << std::endl;
    
    return 0;
}
