#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "npy_loader.h"
#include "split.h"

// 辅助函数: 比较两个4D张量
void compare_4d_tensors(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& cpp_result,
    const std::vector<std::vector<std::vector<std::vector<float>>>>& pytorch_result,
    const std::string& test_name,
    float tolerance = 1e-5
) {
    std::cout << "\n=== 详细数值对比分析 (" << test_name << ") ===" << std::endl;
    
    // 检查维度
    if (cpp_result.size() != pytorch_result.size() ||
        (cpp_result.size() > 0 && cpp_result[0].size() != pytorch_result[0].size()) ||
        (cpp_result.size() > 0 && cpp_result[0].size() > 0 && cpp_result[0][0].size() != pytorch_result[0][0].size()) ||
        (cpp_result.size() > 0 && cpp_result[0].size() > 0 && cpp_result[0][0].size() > 0 && cpp_result[0][0][0].size() != pytorch_result[0][0][0].size())) {
        std::cout << "❌ 维度不匹配!" << std::endl;
        return;
    }
    
    int dim0 = cpp_result.size();
    int dim1 = dim0 > 0 ? cpp_result[0].size() : 0;
    int dim2 = dim1 > 0 ? cpp_result[0][0].size() : 0;
    int dim3 = dim2 > 0 ? cpp_result[0][0][0].size() : 0;
    
    float max_diff = 0.0f;
    int total_elements = dim0 * dim1 * dim2 * dim3;
    int error_count = 0;
    float cpp_sum = 0.0f;
    float pytorch_sum = 0.0f;
    
    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++) {
            for (int k = 0; k < dim2; k++) {
                for (int l = 0; l < dim3; l++) {
                    float cpp_val = cpp_result[i][j][k][l];
                    float pytorch_val = pytorch_result[i][j][k][l];
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
    int show_count = std::min(5, dim3);
    std::cout << "位置[0,0,0,0-" << (show_count-1) << "]:" << std::endl;
    std::cout << "C++:     ";
    for (int l = 0; l < show_count; l++) {
        std::cout << std::setprecision(6) << std::fixed << cpp_result[0][0][0][l] << " ";
    }
    std::cout << std::endl;
    std::cout << "PyTorch: ";
    for (int l = 0; l < show_count; l++) {
        std::cout << std::setprecision(6) << std::fixed << pytorch_result[0][0][0][l] << " ";
    }
    std::cout << std::endl;
    
    if (error_count == 0) {
        std::cout << "\n✅ 测试通过: C++split结果与PyTorch在数值上完全一致!" << std::endl;
    } else {
        std::cout << "\n❌ 测试失败: 存在超出容差的差异!" << std::endl;
    }
}

// 加载split参数的函数
std::vector<int> load_split_params(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open split params file: " + filename);
    }
    
    NpyLoader::skip_npy_header(file);
    
    // 读取参数数组
    std::vector<int> params;
    for (int i = 0; i < 6; i++) {  // 最多6个参数
        int64_t param_int;
        file.read(reinterpret_cast<char*>(&param_int), sizeof(int64_t));
        if (file.eof()) break;
        if (file.fail()) {
            throw std::runtime_error("Failed to read params from file: " + filename);
        }
        params.push_back(static_cast<int>(param_int));
    }
    
    return params;
}

int main() {
    std::cout << "=== Split (4D) 算子详细数值对比测试 ===" << std::endl;
    
    try {
        // 测试案例1: 沿axis=1均等分割 (8->4+4)
        std::cout << "\n=== 测试案例1: 沿axis=1均等分割 (8->4+4) ===" << std::endl;
        
        auto input1 = load_4d_tensor("split_case1_input.npy");
        auto params1 = load_split_params("split_case1_params.npy");
        
        int axis1 = params1[0];
        int num_splits1 = params1[1];
        int split_size1 = params1[2];
        
        std::cout << "输入形状: [" << input1.size() << ", " << input1[0].size() << ", " << input1[0][0].size() << ", " << input1[0][0][0].size() << "]" << std::endl;
        std::cout << "axis: " << axis1 << ", num_splits: " << num_splits1 << ", split_size: " << split_size1 << std::endl;
        
        // C++分割
        auto cpp_results1 = split(input1, axis1, num_splits1);
        std::cout << "C++分割结果数量: " << cpp_results1.size() << std::endl;
        
        // 逐个比较每个分割结果
        for (int i = 0; i < num_splits1; i++) {
            auto pytorch_result = load_4d_tensor("split_case1_output" + std::to_string(i+1) + ".npy");
            std::cout << "\n--- 分割片段 " << (i+1) << " ---" << std::endl;
            std::cout << "C++结果形状: [" << cpp_results1[i].size() << ", " << cpp_results1[i][0].size() << ", " << cpp_results1[i][0][0].size() << ", " << cpp_results1[i][0][0][0].size() << "]" << std::endl;
            std::cout << "PyTorch结果形状: [" << pytorch_result.size() << ", " << pytorch_result[0].size() << ", " << pytorch_result[0][0].size() << ", " << pytorch_result[0][0][0].size() << "]" << std::endl;
            
            compare_4d_tensors(cpp_results1[i], pytorch_result, "均等分割-片段" + std::to_string(i+1));
        }
        
        // 测试案例2: 沿axis=1均等分割 (12->4+4+4)
        std::cout << "\n=== 测试案例2: 沿axis=1均等分割 (12->4+4+4) ===" << std::endl;
        
        auto input2 = load_4d_tensor("split_case2_input.npy");
        auto params2 = load_split_params("split_case2_params.npy");
        
        int axis2 = params2[0];
        int num_splits2 = params2[1];
        int split_size2 = params2[2];
        
        std::cout << "输入形状: [" << input2.size() << ", " << input2[0].size() << ", " << input2[0][0].size() << ", " << input2[0][0][0].size() << "]" << std::endl;
        std::cout << "axis: " << axis2 << ", num_splits: " << num_splits2 << ", split_size: " << split_size2 << std::endl;
        
        auto cpp_results2 = split(input2, axis2, num_splits2);
        std::cout << "C++分割结果数量: " << cpp_results2.size() << std::endl;
        
        for (int i = 0; i < num_splits2; i++) {
            auto pytorch_result = load_4d_tensor("split_case2_output" + std::to_string(i+1) + ".npy");
            std::cout << "\n--- 分割片段 " << (i+1) << " ---" << std::endl;
            
            compare_4d_tensors(cpp_results2[i], pytorch_result, "均等分割-片段" + std::to_string(i+1));
        }
        
        // 测试案例3: 沿axis=1均等分割 (16->4+4+4+4)
        std::cout << "\n=== 测试案例3: 沿axis=1均等分割 (16->4+4+4+4) ===" << std::endl;
        
        auto input3 = load_4d_tensor("split_case3_input.npy");
        auto params3 = load_split_params("split_case3_params.npy");
        
        int axis3 = params3[0];
        int num_splits3 = params3[1];
        int split_size3 = params3[2];
        
        std::cout << "输入形状: [" << input3.size() << ", " << input3[0].size() << ", " << input3[0][0].size() << ", " << input3[0][0][0].size() << "]" << std::endl;
        std::cout << "axis: " << axis3 << ", num_splits: " << num_splits3 << ", split_size: " << split_size3 << std::endl;
        
        auto cpp_results3 = split(input3, axis3, num_splits3);
        std::cout << "C++分割结果数量: " << cpp_results3.size() << std::endl;
        
        for (int i = 0; i < num_splits3; i++) {
            auto pytorch_result = load_4d_tensor("split_case3_output" + std::to_string(i+1) + ".npy");
            std::cout << "\n--- 分割片段 " << (i+1) << " ---" << std::endl;
            
            compare_4d_tensors(cpp_results3[i], pytorch_result, "均等分割-片段" + std::to_string(i+1));
        }
        
        // 测试案例4: 不均等分割 (10->3+4+3)
        std::cout << "\n=== 测试案例4: 沿axis=1不均等分割 (10->3+4+3) ===" << std::endl;
        
        auto input4 = load_4d_tensor("split_case4_input.npy");
        auto params4 = load_split_params("split_case4_params.npy");
        
        int axis4 = params4[0];
        std::vector<int> split_sizes4(params4.begin() + 1, params4.end());
        
        std::cout << "输入形状: [" << input4.size() << ", " << input4[0].size() << ", " << input4[0][0].size() << ", " << input4[0][0][0].size() << "]" << std::endl;
        std::cout << "axis: " << axis4 << ", split_sizes: [";
        for (size_t i = 0; i < split_sizes4.size(); i++) {
            std::cout << split_sizes4[i];
            if (i < split_sizes4.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        auto cpp_results4 = split(input4, axis4, 0, split_sizes4);
        std::cout << "C++分割结果数量: " << cpp_results4.size() << std::endl;
        
        for (size_t i = 0; i < split_sizes4.size(); i++) {
            auto pytorch_result = load_4d_tensor("split_case4_output" + std::to_string(i+1) + ".npy");
            std::cout << "\n--- 分割片段 " << (i+1) << " ---" << std::endl;
            
            compare_4d_tensors(cpp_results4[i], pytorch_result, "不均等分割-片段" + std::to_string(i+1));
        }
        
        // 测试案例5: 边界测试 (6->3+3)
        std::cout << "\n=== 测试案例5: 沿axis=1分成2份 (6->3+3) ===" << std::endl;
        
        auto input5 = load_4d_tensor("split_case5_input.npy");
        auto params5 = load_split_params("split_case5_params.npy");
        
        int axis5 = params5[0];
        int num_splits5 = params5[1];
        int split_size5 = params5[2];
        
        std::cout << "输入形状: [" << input5.size() << ", " << input5[0].size() << ", " << input5[0][0].size() << ", " << input5[0][0][0].size() << "]" << std::endl;
        std::cout << "axis: " << axis5 << ", num_splits: " << num_splits5 << ", split_size: " << split_size5 << std::endl;
        
        auto cpp_results5 = split(input5, axis5, num_splits5);
        std::cout << "C++分割结果数量: " << cpp_results5.size() << std::endl;
        
        for (int i = 0; i < num_splits5; i++) {
            auto pytorch_result = load_4d_tensor("split_case5_output" + std::to_string(i+1) + ".npy");
            std::cout << "\n--- 分割片段 " << (i+1) << " ---" << std::endl;
            
            compare_4d_tensors(cpp_results5[i], pytorch_result, "边界测试-片段" + std::to_string(i+1));
        }
        
        std::cout << "\n=== Split (4D) 测试完成 ===" << std::endl;
        std::cout << "注意: 测试进行了逐元素的数值对比，确保C++split实现与PyTorch的数学一致性" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "测试过程中发生错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
