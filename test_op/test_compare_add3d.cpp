#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>
#include "add_3d.h"

// 从npy文件加载3D张量 (简化版本，只支持float32)
std::vector<std::vector<std::vector<float>>> load_npy_3d(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // 简化处理：跳过numpy头部
    char header[1024];
    file.read(header, 128);
    
    // 根据文件名确定维度
    std::vector<std::vector<std::vector<float>>> tensor;
    
    if (filename.find("test1_input1") != std::string::npos || filename.find("test1_input2") != std::string::npos || filename.find("test1_result") != std::string::npos) {
        tensor.resize(3, std::vector<std::vector<float>>(4, std::vector<float>(5)));
    } else if (filename.find("test2_input1") != std::string::npos || filename.find("test2_result") != std::string::npos) {
        tensor.resize(3, std::vector<std::vector<float>>(4, std::vector<float>(5)));
    } else if (filename.find("test2_input2") != std::string::npos) {
        tensor.resize(1, std::vector<std::vector<float>>(4, std::vector<float>(5)));
    } else if (filename.find("test3_input1") != std::string::npos || filename.find("test3_result") != std::string::npos) {
        tensor.resize(2, std::vector<std::vector<float>>(3, std::vector<float>(4)));
    } else if (filename.find("test3_input2") != std::string::npos) {
        tensor.resize(2, std::vector<std::vector<float>>(1, std::vector<float>(4)));
    }
    
    // 读取数据
    for (int d = 0; d < tensor.size(); d++) {
        for (int h = 0; h < tensor[0].size(); h++) {
            for (int w = 0; w < tensor[0][0].size(); w++) {
                file.read(reinterpret_cast<char*>(&tensor[d][h][w]), sizeof(float));
            }
        }
    }
    
    return tensor;
}

// 比较两个3D张量
bool compare_tensors_3d(const std::vector<std::vector<std::vector<float>>>& tensor1,
                       const std::vector<std::vector<std::vector<float>>>& tensor2,
                       float tolerance = 1e-5) {
    
    if (tensor1.size() != tensor2.size()) return false;
    if (tensor1.empty()) return true;
    if (tensor1[0].size() != tensor2[0].size()) return false;
    if (tensor1[0].empty()) return true;
    if (tensor1[0][0].size() != tensor2[0][0].size()) return false;
    
    int diff_count = 0;
    float max_diff = 0.0f;
    
    for (int d = 0; d < tensor1.size(); d++) {
        for (int h = 0; h < tensor1[0].size(); h++) {
            for (int w = 0; w < tensor1[0][0].size(); w++) {
                float diff = std::abs(tensor1[d][h][w] - tensor2[d][h][w]);
                max_diff = std::max(max_diff, diff);
                if (diff > tolerance) {
                    diff_count++;
                    if (diff_count <= 5) {
                        std::cout << "差异 [" << d << "," << h << "," << w << "]: " 
                                 << tensor1[d][h][w] << " vs " << tensor2[d][h][w] 
                                 << " (diff: " << diff << ")" << std::endl;
                    }
                }
            }
        }
    }
    
    std::cout << "最大差异: " << max_diff << std::endl;
    std::cout << "超出容差的元素数: " << diff_count << std::endl;
    
    return diff_count == 0;
}

// 打印3D张量信息
void print_tensor_info_3d(const std::vector<std::vector<std::vector<float>>>& tensor, 
                         const std::string& name) {
    std::cout << name << " shape: [" << tensor.size() << ", " 
              << (tensor.size() > 0 ? tensor[0].size() : 0) << ", " 
              << (tensor.size() > 0 && tensor[0].size() > 0 ? tensor[0][0].size() : 0) << "]" << std::endl;
}

int main() {
    std::cout << "=== add_3d C++与PyTorch结果直接对比测试 ===" << std::endl << std::endl;
    
    try {
        // 测试案例1: 基本加法
        std::cout << "测试案例1: 基本3D张量加法" << std::endl;
        
        auto input1 = load_npy_3d("add3d_test1_input1.npy");
        auto input2 = load_npy_3d("add3d_test1_input2.npy");
        auto pytorch_result = load_npy_3d("add3d_test1_result_pytorch.npy");
        
        print_tensor_info_3d(input1, "输入1");
        print_tensor_info_3d(input2, "输入2");
        print_tensor_info_3d(pytorch_result, "PyTorch结果");
        
        auto cpp_result = add_3d(input1, input2);
        print_tensor_info_3d(cpp_result, "C++结果");
        
        std::cout << "\n结果对比:" << std::endl;
        bool match1 = compare_tensors_3d(cpp_result, pytorch_result);
        std::cout << "测试1结果: " << (match1 ? "✅ 完全匹配" : "❌ 存在差异") << std::endl;
        
        // 显示部分数值
        std::cout << "\n部分数值对比:" << std::endl;
        std::cout << "位置[0,0,0-2]:" << std::endl;
        std::cout << "C++:     ";
        for (int i = 0; i < 3; i++) {
            std::cout << std::fixed << std::setprecision(6) << cpp_result[0][0][i] << " ";
        }
        std::cout << std::endl;
        std::cout << "PyTorch: ";
        for (int i = 0; i < 3; i++) {
            std::cout << std::fixed << std::setprecision(6) << pytorch_result[0][0][i] << " ";
        }
        std::cout << std::endl << std::endl;
        
        // 测试案例2: 广播
        std::cout << "测试案例2: 广播操作" << std::endl;
        
        auto input3 = load_npy_3d("add3d_test2_input1.npy");
        auto input4 = load_npy_3d("add3d_test2_input2.npy");
        auto pytorch_result2 = load_npy_3d("add3d_test2_result_pytorch.npy");
        
        print_tensor_info_3d(input3, "输入1");
        print_tensor_info_3d(input4, "输入2");
        print_tensor_info_3d(pytorch_result2, "PyTorch结果");
        
        auto cpp_result2 = add_3d(input3, input4);
        print_tensor_info_3d(cpp_result2, "C++结果");
        
        std::cout << "\n结果对比:" << std::endl;
        bool match2 = compare_tensors_3d(cpp_result2, pytorch_result2);
        std::cout << "测试2结果: " << (match2 ? "✅ 完全匹配" : "❌ 存在差异") << std::endl << std::endl;
        
        // 测试案例3: 复杂广播
        std::cout << "测试案例3: 复杂广播" << std::endl;
        
        auto input5 = load_npy_3d("add3d_test3_input1.npy");
        auto input6 = load_npy_3d("add3d_test3_input2.npy");
        auto pytorch_result3 = load_npy_3d("add3d_test3_result_pytorch.npy");
        
        print_tensor_info_3d(input5, "输入1");
        print_tensor_info_3d(input6, "输入2");
        print_tensor_info_3d(pytorch_result3, "PyTorch结果");
        
        auto cpp_result3 = add_3d(input5, input6);
        print_tensor_info_3d(cpp_result3, "C++结果");
        
        std::cout << "\n结果对比:" << std::endl;
        bool match3 = compare_tensors_3d(cpp_result3, pytorch_result3);
        std::cout << "测试3结果: " << (match3 ? "✅ 完全匹配" : "❌ 存在差异") << std::endl;
        
        std::cout << std::endl << "=== add_3d对比测试完成 ===" << std::endl;
        std::cout << "总体结果: " << ((match1 && match2 && match3) ? "✅ 所有测试通过" : "❌ 存在差异") << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
