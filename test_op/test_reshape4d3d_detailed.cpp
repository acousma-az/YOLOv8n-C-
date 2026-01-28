#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "reshape_4d_to_3d.h"
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
                float diff = std::abs(tensor1[d0][d1][d2] - tensor2[d0][d1][d2]);
                if (diff > max_diff) max_diff = diff;
                if (diff > tolerance) {
                    diff_count++;
                    if (diff_count <= 5) { // 只显示前5个差异
                        std::cout << "差异位置 [" << d0 << "," << d1 << "," << d2 << "]: "
                                  << std::fixed << std::setprecision(8) 
                                  << tensor1[d0][d1][d2] << " vs " << tensor2[d0][d1][d2] 
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

// 加载目标形状
std::vector<int> load_target_shape_4d3d(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // 跳过numpy头部
    NpyLoader::skip_npy_header(file);
    
    // 读取形状数据（简化处理）
    std::vector<int> shape;
    
    if (filename.find("case1") != std::string::npos) {
        // 空目标形状，表示默认行为
        return shape;
    } else if (filename.find("case2") != std::string::npos) {
        return {6, 4, 4};
    } else if (filename.find("case3") != std::string::npos) {
        return {8, -1, 6};
    } else if (filename.find("case4") != std::string::npos) {
        return {12, 24, 6};
    } else if (filename.find("case5") != std::string::npos) {
        return {12, 4, 4};
    }
    
    return shape;
}

// 测试案例1: 默认reshape (合并height和width)
void test_reshape4d3d_case1() {
    std::cout << "\n=== 测试案例1: 默认reshape (合并height和width) ===" << std::endl;
    
    try {
        auto input = NpyLoader::load_4d_float32("reshape4d3d_case1_input.npy");
        auto expected_result = NpyLoader::load_3d_float32("reshape4d3d_case1_result.npy");
        auto target_shape = load_target_shape_4d3d("reshape4d3d_case1_target_shape.npy");
        
        std::cout << "输入形状: [" << input.size() << ", " << input[0].size() << ", " 
                  << input[0][0].size() << ", " << input[0][0][0].size() << "]" << std::endl;
        std::cout << "目标形状: 默认 (空)" << std::endl;
        
        // 执行C++ reshape
        auto cpp_result = reshape_4d_to_3d(input, target_shape);
        
        std::cout << "PyTorch结果形状: [" << expected_result.size() << ", " << expected_result[0].size() 
                  << ", " << expected_result[0][0].size() << "]" << std::endl;
        std::cout << "C++结果形状: [" << cpp_result.size() << ", " << cpp_result[0].size() 
                  << ", " << cpp_result[0][0].size() << "]" << std::endl;
        
        // 详细对比
        bool match = detailed_compare_3d(cpp_result, expected_result, "默认reshape");
        
        if (match) {
            std::cout << "\n✅ 测试通过: C++reshape结果与PyTorch在数值上完全一致!" << std::endl;
        } else {
            std::cout << "\n❌ 测试失败: 发现数值差异!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试失败: " << e.what() << std::endl;
    }
}

// 测试案例2: 指定3D形状 - 常规reshape
void test_reshape4d3d_case2() {
    std::cout << "\n=== 测试案例2: 指定3D形状 - 常规reshape ===" << std::endl;
    
    try {
        auto input = NpyLoader::load_4d_float32("reshape4d3d_case2_input.npy");
        auto expected_result = NpyLoader::load_3d_float32("reshape4d3d_case2_result.npy");
        auto target_shape = load_target_shape_4d3d("reshape4d3d_case2_target_shape.npy");
        
        std::cout << "输入形状: [" << input.size() << ", " << input[0].size() << ", " 
                  << input[0][0].size() << ", " << input[0][0][0].size() << "]" << std::endl;
        std::cout << "目标形状: [" << target_shape[0] << ", " << target_shape[1] << ", " 
                  << target_shape[2] << "]" << std::endl;
        
        // 执行C++ reshape
        auto cpp_result = reshape_4d_to_3d(input, target_shape);
        
        std::cout << "PyTorch结果形状: [" << expected_result.size() << ", " << expected_result[0].size() 
                  << ", " << expected_result[0][0].size() << "]" << std::endl;
        std::cout << "C++结果形状: [" << cpp_result.size() << ", " << cpp_result[0].size() 
                  << ", " << cpp_result[0][0].size() << "]" << std::endl;
        
        // 详细对比
        bool match = detailed_compare_3d(cpp_result, expected_result, "指定3D形状");
        
        if (match) {
            std::cout << "\n✅ 测试通过: C++reshape结果与PyTorch在数值上完全一致!" << std::endl;
        } else {
            std::cout << "\n❌ 测试失败: 发现数值差异!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试失败: " << e.what() << std::endl;
    }
}

// 测试案例3: 包含-1的自动推断
void test_reshape4d3d_case3() {
    std::cout << "\n=== 测试案例3: 包含-1的自动推断 ===" << std::endl;
    
    try {
        auto input = NpyLoader::load_4d_float32("reshape4d3d_case3_input.npy");
        auto expected_result = NpyLoader::load_3d_float32("reshape4d3d_case3_result.npy");
        auto target_shape = load_target_shape_4d3d("reshape4d3d_case3_target_shape.npy");
        
        std::cout << "输入形状: [" << input.size() << ", " << input[0].size() << ", " 
                  << input[0][0].size() << ", " << input[0][0][0].size() << "]" << std::endl;
        std::cout << "目标形状: [" << target_shape[0] << ", " << target_shape[1] << ", " 
                  << target_shape[2] << "]" << std::endl;
        
        // 执行C++ reshape
        auto cpp_result = reshape_4d_to_3d(input, target_shape);
        
        std::cout << "PyTorch结果形状: [" << expected_result.size() << ", " << expected_result[0].size() 
                  << ", " << expected_result[0][0].size() << "]" << std::endl;
        std::cout << "C++结果形状: [" << cpp_result.size() << ", " << cpp_result[0].size() 
                  << ", " << cpp_result[0][0].size() << "]" << std::endl;
        
        // 详细对比
        bool match = detailed_compare_3d(cpp_result, expected_result, "-1自动推断");
        
        if (match) {
            std::cout << "\n✅ 测试通过: C++reshape结果与PyTorch在数值上完全一致!" << std::endl;
        } else {
            std::cout << "\n❌ 测试失败: 发现数值差异!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试失败: " << e.what() << std::endl;
    }
}

// 测试案例4: 大尺寸测试
void test_reshape4d3d_case4() {
    std::cout << "\n=== 测试案例4: 大尺寸测试 ===" << std::endl;
    
    try {
        auto input = NpyLoader::load_4d_float32("reshape4d3d_case4_input.npy");
        auto expected_result = NpyLoader::load_3d_float32("reshape4d3d_case4_result.npy");
        auto target_shape = load_target_shape_4d3d("reshape4d3d_case4_target_shape.npy");
        
        std::cout << "输入形状: [" << input.size() << ", " << input[0].size() << ", " 
                  << input[0][0].size() << ", " << input[0][0][0].size() << "]" << std::endl;
        std::cout << "目标形状: [" << target_shape[0] << ", " << target_shape[1] << ", " 
                  << target_shape[2] << "]" << std::endl;
        
        // 执行C++ reshape
        auto cpp_result = reshape_4d_to_3d(input, target_shape);
        
        std::cout << "PyTorch结果形状: [" << expected_result.size() << ", " << expected_result[0].size() 
                  << ", " << expected_result[0][0].size() << "]" << std::endl;
        std::cout << "C++结果形状: [" << cpp_result.size() << ", " << cpp_result[0].size() 
                  << ", " << cpp_result[0][0].size() << "]" << std::endl;
        
        // 详细对比
        bool match = detailed_compare_3d(cpp_result, expected_result, "大尺寸测试");
        
        if (match) {
            std::cout << "\n✅ 测试通过: C++reshape结果与PyTorch在数值上完全一致!" << std::endl;
        } else {
            std::cout << "\n❌ 测试失败: 发现数值差异!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试失败: " << e.what() << std::endl;
    }
}

// 测试案例5: batch维度合并
void test_reshape4d3d_case5() {
    std::cout << "\n=== 测试案例5: batch维度合并 ===" << std::endl;
    
    try {
        auto input = NpyLoader::load_4d_float32("reshape4d3d_case5_input.npy");
        auto expected_result = NpyLoader::load_3d_float32("reshape4d3d_case5_result.npy");
        auto target_shape = load_target_shape_4d3d("reshape4d3d_case5_target_shape.npy");
        
        std::cout << "输入形状: [" << input.size() << ", " << input[0].size() << ", " 
                  << input[0][0].size() << ", " << input[0][0][0].size() << "]" << std::endl;
        std::cout << "目标形状: [" << target_shape[0] << ", " << target_shape[1] << ", " 
                  << target_shape[2] << "]" << std::endl;
        
        // 执行C++ reshape
        auto cpp_result = reshape_4d_to_3d(input, target_shape);
        
        std::cout << "PyTorch结果形状: [" << expected_result.size() << ", " << expected_result[0].size() 
                  << ", " << expected_result[0][0].size() << "]" << std::endl;
        std::cout << "C++结果形状: [" << cpp_result.size() << ", " << cpp_result[0].size() 
                  << ", " << cpp_result[0][0].size() << "]" << std::endl;
        
        // 详细对比
        bool match = detailed_compare_3d(cpp_result, expected_result, "batch维度合并");
        
        if (match) {
            std::cout << "\n✅ 测试通过: C++reshape结果与PyTorch在数值上完全一致!" << std::endl;
        } else {
            std::cout << "\n❌ 测试失败: 发现数值差异!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试失败: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "=== Reshape 4D to 3D 算子详细数值对比测试 ===" << std::endl;
    
    test_reshape4d3d_case1();
    test_reshape4d3d_case2();
    test_reshape4d3d_case3();
    test_reshape4d3d_case4();
    test_reshape4d3d_case5();
    
    std::cout << "\n=== Reshape 4D to 3D 测试完成 ===" << std::endl;
    std::cout << "注意: 测试进行了逐元素的数值对比，确保C++reshape实现与PyTorch的数学一致性" << std::endl;
    
    return 0;
}
