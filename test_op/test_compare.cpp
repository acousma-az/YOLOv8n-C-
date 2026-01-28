#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>
#include "concat.h"

// 从npy文件加载4D张量 (简化版本，只支持float32)
std::vector<std::vector<std::vector<std::vector<float>>>> load_npy_4d(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // 跳过numpy头部 (简化处理)
    char header[1024];
    file.read(header, 128);
    
    // 手动指定维度 (简化版本)
    std::vector<std::vector<std::vector<std::vector<float>>>> tensor;
    
    if (filename.find("test1_input1") != std::string::npos) {
        tensor.resize(2, std::vector<std::vector<std::vector<float>>>(3, 
            std::vector<std::vector<float>>(4, std::vector<float>(4))));
    } else if (filename.find("test1_input2") != std::string::npos) {
        tensor.resize(2, std::vector<std::vector<std::vector<float>>>(2, 
            std::vector<std::vector<float>>(4, std::vector<float>(4))));
    } else if (filename.find("test1_input3") != std::string::npos) {
        tensor.resize(2, std::vector<std::vector<std::vector<float>>>(1, 
            std::vector<std::vector<float>>(4, std::vector<float>(4))));
    } else if (filename.find("test1_result") != std::string::npos) {
        tensor.resize(2, std::vector<std::vector<std::vector<float>>>(6, 
            std::vector<std::vector<float>>(4, std::vector<float>(4))));
    } else if (filename.find("test2_input1") != std::string::npos) {
        tensor.resize(1, std::vector<std::vector<std::vector<float>>>(8, 
            std::vector<std::vector<float>>(6, std::vector<float>(6))));
    } else if (filename.find("test2_input2") != std::string::npos) {
        tensor.resize(1, std::vector<std::vector<std::vector<float>>>(4, 
            std::vector<std::vector<float>>(6, std::vector<float>(6))));
    } else if (filename.find("test2_result") != std::string::npos) {
        tensor.resize(1, std::vector<std::vector<std::vector<float>>>(12, 
            std::vector<std::vector<float>>(6, std::vector<float>(6))));
    }
    
    // 读取数据
    for (int b = 0; b < tensor.size(); b++) {
        for (int c = 0; c < tensor[0].size(); c++) {
            for (int h = 0; h < tensor[0][0].size(); h++) {
                for (int w = 0; w < tensor[0][0][0].size(); w++) {
                    file.read(reinterpret_cast<char*>(&tensor[b][c][h][w]), sizeof(float));
                }
            }
        }
    }
    
    return tensor;
}

// 比较两个4D张量
bool compare_tensors(const std::vector<std::vector<std::vector<std::vector<float>>>>& tensor1,
                    const std::vector<std::vector<std::vector<std::vector<float>>>>& tensor2,
                    float tolerance = 1e-5) {
    
    if (tensor1.size() != tensor2.size()) return false;
    if (tensor1.empty()) return true;
    if (tensor1[0].size() != tensor2[0].size()) return false;
    if (tensor1[0].empty()) return true;
    if (tensor1[0][0].size() != tensor2[0][0].size()) return false;
    if (tensor1[0][0].empty()) return true;
    if (tensor1[0][0][0].size() != tensor2[0][0][0].size()) return false;
    
    int diff_count = 0;
    float max_diff = 0.0f;
    
    for (int b = 0; b < tensor1.size(); b++) {
        for (int c = 0; c < tensor1[0].size(); c++) {
            for (int h = 0; h < tensor1[0][0].size(); h++) {
                for (int w = 0; w < tensor1[0][0][0].size(); w++) {
                    float diff = std::abs(tensor1[b][c][h][w] - tensor2[b][c][h][w]);
                    max_diff = std::max(max_diff, diff);
                    if (diff > tolerance) {
                        diff_count++;
                        if (diff_count <= 5) { // 只打印前5个差异
                            std::cout << "差异 [" << b << "," << c << "," << h << "," << w << "]: " 
                                     << tensor1[b][c][h][w] << " vs " << tensor2[b][c][h][w] 
                                     << " (diff: " << diff << ")" << std::endl;
                        }
                    }
                }
            }
        }
    }
    
    std::cout << "最大差异: " << max_diff << std::endl;
    std::cout << "超出容差的元素数: " << diff_count << std::endl;
    
    return diff_count == 0;
}

// 打印张量信息
void print_tensor_info(const std::vector<std::vector<std::vector<std::vector<float>>>>& tensor, 
                       const std::string& name) {
    std::cout << name << " shape: [" << tensor.size() << ", " 
              << (tensor.size() > 0 ? tensor[0].size() : 0) << ", " 
              << (tensor.size() > 0 && tensor[0].size() > 0 ? tensor[0][0].size() : 0) << ", "
              << (tensor.size() > 0 && tensor[0].size() > 0 && tensor[0][0].size() > 0 ? tensor[0][0][0].size() : 0) << "]" << std::endl;
}

int main() {
    std::cout << "=== C++与PyTorch结果直接对比测试 ===" << std::endl << std::endl;
    
    try {
        // 测试案例1
        std::cout << "测试案例1: 基本连接对比" << std::endl;
        
        // 加载输入数据
        auto input1 = load_npy_4d("test1_input1.npy");
        auto input2 = load_npy_4d("test1_input2.npy");
        auto input3 = load_npy_4d("test1_input3.npy");
        auto pytorch_result = load_npy_4d("test1_result_pytorch.npy");
        
        print_tensor_info(input1, "输入1");
        print_tensor_info(input2, "输入2");
        print_tensor_info(input3, "输入3");
        print_tensor_info(pytorch_result, "PyTorch结果");
        
        // C++计算
        std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> inputs = {input1, input2, input3};
        auto cpp_result = concat(inputs, 1);
        
        print_tensor_info(cpp_result, "C++结果");
        
        // 对比结果
        std::cout << "\n结果对比:" << std::endl;
        bool match = compare_tensors(cpp_result, pytorch_result);
        std::cout << "测试1结果: " << (match ? "✅ 完全匹配" : "❌ 存在差异") << std::endl;
        
        // 显示部分数值
        std::cout << "\n部分数值对比:" << std::endl;
        std::cout << "位置[0,0,0,0-2]:" << std::endl;
        std::cout << "C++:     ";
        for (int i = 0; i < 3; i++) {
            std::cout << std::fixed << std::setprecision(6) << cpp_result[0][0][0][i] << " ";
        }
        std::cout << std::endl;
        std::cout << "PyTorch: ";
        for (int i = 0; i < 3; i++) {
            std::cout << std::fixed << std::setprecision(6) << pytorch_result[0][0][0][i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "\n位置[0,3,0,0-2] (第二个tensor的第一个通道):" << std::endl;
        std::cout << "C++:     ";
        for (int i = 0; i < 3; i++) {
            std::cout << std::fixed << std::setprecision(6) << cpp_result[0][3][0][i] << " ";
        }
        std::cout << std::endl;
        std::cout << "PyTorch: ";
        for (int i = 0; i < 3; i++) {
            std::cout << std::fixed << std::setprecision(6) << pytorch_result[0][3][0][i] << " ";
        }
        std::cout << std::endl << std::endl;
        
        // 测试案例2
        std::cout << "测试案例2: 不同通道数对比" << std::endl;
        
        auto input4 = load_npy_4d("test2_input1.npy");
        auto input5 = load_npy_4d("test2_input2.npy");
        auto pytorch_result2 = load_npy_4d("test2_result_pytorch.npy");
        
        print_tensor_info(input4, "输入1");
        print_tensor_info(input5, "输入2");
        print_tensor_info(pytorch_result2, "PyTorch结果");
        
        std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> inputs2 = {input4, input5};
        auto cpp_result2 = concat(inputs2, 1);
        
        print_tensor_info(cpp_result2, "C++结果");
        
        std::cout << "\n结果对比:" << std::endl;
        bool match2 = compare_tensors(cpp_result2, pytorch_result2);
        std::cout << "测试2结果: " << (match2 ? "✅ 完全匹配" : "❌ 存在差异") << std::endl;
        
        std::cout << std::endl << "=== 对比测试完成 ===" << std::endl;
        std::cout << "总体结果: " << ((match && match2) ? "✅ 所有测试通过" : "❌ 存在差异") << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
