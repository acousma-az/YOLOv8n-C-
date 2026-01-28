#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "npy_loader.h"
#include "softmax.h"

// 辅助函数: 比较两个4D张量
void compare_4d_tensors(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& cpp_result,
    const std::vector<std::vector<std::vector<std::vector<float>>>>& pytorch_result,
    const std::string& test_name,
    int axis,
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
    
    // 验证softmax性质：沿指定轴的和应该为1
    std::cout << "\n=== Softmax性质验证 ===" << std::endl;
    float sum_along_axis = 0.0f;
    int count = 0;
    
    if (axis == 0) {
        for (int j = 0; j < dim1; j++) {
            for (int k = 0; k < dim2; k++) {
                for (int l = 0; l < dim3; l++) {
                    float slice_sum = 0.0f;
                    for (int i = 0; i < dim0; i++) {
                        slice_sum += cpp_result[i][j][k][l];
                    }
                    sum_along_axis += slice_sum;
                    count++;
                }
            }
        }
    } else if (axis == 1) {
        for (int i = 0; i < dim0; i++) {
            for (int k = 0; k < dim2; k++) {
                for (int l = 0; l < dim3; l++) {
                    float slice_sum = 0.0f;
                    for (int j = 0; j < dim1; j++) {
                        slice_sum += cpp_result[i][j][k][l];
                    }
                    sum_along_axis += slice_sum;
                    count++;
                }
            }
        }
    } else if (axis == 2) {
        for (int i = 0; i < dim0; i++) {
            for (int j = 0; j < dim1; j++) {
                for (int l = 0; l < dim3; l++) {
                    float slice_sum = 0.0f;
                    for (int k = 0; k < dim2; k++) {
                        slice_sum += cpp_result[i][j][k][l];
                    }
                    sum_along_axis += slice_sum;
                    count++;
                }
            }
        }
    } else { // axis == 3
        for (int i = 0; i < dim0; i++) {
            for (int j = 0; j < dim1; j++) {
                for (int k = 0; k < dim2; k++) {
                    float slice_sum = 0.0f;
                    for (int l = 0; l < dim3; l++) {
                        slice_sum += cpp_result[i][j][k][l];
                    }
                    sum_along_axis += slice_sum;
                    count++;
                }
            }
        }
    }
    
    float avg_sum = sum_along_axis / count;
    std::cout << "沿axis=" << axis << "的平均总和: " << avg_sum << std::endl;
    std::cout << "期望总和: 1.0" << std::endl;
    std::cout << "总和差异: " << std::abs(avg_sum - 1.0f) << std::endl;
    
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
    
    if (error_count == 0 && std::abs(avg_sum - 1.0f) < tolerance) {
        std::cout << "\n✅ 测试通过: C++softmax结果与PyTorch在数值上完全一致，且满足softmax性质!" << std::endl;
    } else {
        std::cout << "\n❌ 测试失败: 存在超出容差的差异或不满足softmax性质!" << std::endl;
    }
}

// 加载axis参数的函数
int load_axis_param(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open axis params file: " + filename);
    }
    
    NpyLoader::skip_npy_header(file);
    
    // 读取axis参数 (int64)
    int64_t axis_int;
    file.read(reinterpret_cast<char*>(&axis_int), sizeof(int64_t));
    if (file.fail()) {
        throw std::runtime_error("Failed to read axis from file: " + filename);
    }
    
    return static_cast<int>(axis_int);
}

int main() {
    std::cout << "=== Softmax 算子详细数值对比测试 ===" << std::endl;
    
    try {
        // 测试案例1: 沿axis=0的softmax
        std::cout << "\n=== 测试案例1: 沿axis=0的softmax ===" << std::endl;
        
        auto input1 = load_4d_tensor("softmax_case1_input.npy");
        auto pytorch_result1 = load_4d_tensor("softmax_case1_result.npy");
        int axis1 = load_axis_param("softmax_case1_params.npy");
        
        std::cout << "输入形状: [" << input1.size() << ", " << input1[0].size() << ", " << input1[0][0].size() << ", " << input1[0][0][0].size() << "]" << std::endl;
        std::cout << "axis: " << axis1 << std::endl;
        std::cout << "PyTorch结果形状: [" << pytorch_result1.size() << ", " << pytorch_result1[0].size() << ", " << pytorch_result1[0][0].size() << ", " << pytorch_result1[0][0][0].size() << "]" << std::endl;
        
        auto cpp_result1 = softmax(input1, axis1);
        std::cout << "C++结果形状: [" << cpp_result1.size() << ", " << cpp_result1[0].size() << ", " << cpp_result1[0][0].size() << ", " << cpp_result1[0][0][0].size() << "]" << std::endl;
        
        compare_4d_tensors(cpp_result1, pytorch_result1, "沿axis=0的softmax", axis1);
        
        // 测试案例2: 沿axis=1的softmax
        std::cout << "\n=== 测试案例2: 沿axis=1的softmax ===" << std::endl;
        
        auto input2 = load_4d_tensor("softmax_case2_input.npy");
        auto pytorch_result2 = load_4d_tensor("softmax_case2_result.npy");
        int axis2 = load_axis_param("softmax_case2_params.npy");
        
        std::cout << "输入形状: [" << input2.size() << ", " << input2[0].size() << ", " << input2[0][0].size() << ", " << input2[0][0][0].size() << "]" << std::endl;
        std::cout << "axis: " << axis2 << std::endl;
        
        auto cpp_result2 = softmax(input2, axis2);
        
        compare_4d_tensors(cpp_result2, pytorch_result2, "沿axis=1的softmax", axis2);
        
        // 测试案例3: 沿axis=2的softmax
        std::cout << "\n=== 测试案例3: 沿axis=2的softmax ===" << std::endl;
        
        auto input3 = load_4d_tensor("softmax_case3_input.npy");
        auto pytorch_result3 = load_4d_tensor("softmax_case3_result.npy");
        int axis3 = load_axis_param("softmax_case3_params.npy");
        
        std::cout << "输入形状: [" << input3.size() << ", " << input3[0].size() << ", " << input3[0][0].size() << ", " << input3[0][0][0].size() << "]" << std::endl;
        std::cout << "axis: " << axis3 << std::endl;
        
        auto cpp_result3 = softmax(input3, axis3);
        
        compare_4d_tensors(cpp_result3, pytorch_result3, "沿axis=2的softmax", axis3);
        
        // 测试案例4: 沿axis=3的softmax
        std::cout << "\n=== 测试案例4: 沿axis=3的softmax ===" << std::endl;
        
        auto input4 = load_4d_tensor("softmax_case4_input.npy");
        auto pytorch_result4 = load_4d_tensor("softmax_case4_result.npy");
        int axis4 = load_axis_param("softmax_case4_params.npy");
        
        std::cout << "输入形状: [" << input4.size() << ", " << input4[0].size() << ", " << input4[0][0].size() << ", " << input4[0][0][0].size() << "]" << std::endl;
        std::cout << "axis: " << axis4 << std::endl;
        
        auto cpp_result4 = softmax(input4, axis4);
        
        compare_4d_tensors(cpp_result4, pytorch_result4, "沿axis=3的softmax", axis4);
        
        // 测试案例5: 数值稳定性测试
        std::cout << "\n=== 测试案例5: 数值稳定性测试 ===" << std::endl;
        
        auto input5 = load_4d_tensor("softmax_case5_input.npy");
        auto pytorch_result5 = load_4d_tensor("softmax_case5_result.npy");
        int axis5 = load_axis_param("softmax_case5_params.npy");
        
        std::cout << "输入形状: [" << input5.size() << ", " << input5[0].size() << ", " << input5[0][0].size() << ", " << input5[0][0][0].size() << "]" << std::endl;
        std::cout << "axis: " << axis5 << std::endl;
        
        auto cpp_result5 = softmax(input5, axis5);
        
        compare_4d_tensors(cpp_result5, pytorch_result5, "数值稳定性测试", axis5);
        
        std::cout << "\n=== Softmax 测试完成 ===" << std::endl;
        std::cout << "注意: 测试进行了逐元素的数值对比，并验证了softmax的数学性质" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "测试过程中发生错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
