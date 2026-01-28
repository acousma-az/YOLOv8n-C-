import torch
import numpy as np

def save_4d_tensor(tensor, filename):
    """保存4D张量到npy文件"""
    numpy_data = tensor.detach().cpu().numpy()
    np.save(filename, numpy_data)
    print(f"保存 {filename}: {tensor.shape}")

def test_transpose():
    """生成transpose测试数据"""
    print("=== 生成 Transpose 对比测试数据 ===\n")
    
    # 测试案例1: 基础转置 [0, 2, 1, 3]
    print("生成测试案例1: 基础转置 [0, 2, 1, 3]")
    input1 = torch.randn(2, 3, 4, 5)
    perm1 = [0, 2, 1, 3]
    result1 = input1.permute(*perm1)
    
    print(f"输入形状: {input1.shape}")
    print(f"转置轴: {perm1}")
    print(f"结果形状: {result1.shape}")
    print(f"输入范围: [{input1.min():.6f}, {input1.max():.6f}]")
    print(f"结果范围: [{result1.min():.6f}, {result1.max():.6f}]")
    
    save_4d_tensor(input1, "transpose_case1_input.npy")
    save_4d_tensor(result1, "transpose_case1_result.npy")
    
    # 保存转置参数
    perm_params1 = np.array(perm1)
    np.save("transpose_case1_params.npy", perm_params1)
    print(f"保存参数: {perm1}")
    print("保存到: transpose_case1_*.npy\n")
    
    # 测试案例2: 小尺寸张量
    print("生成测试案例2: 小尺寸张量")
    input2 = torch.randn(1, 2, 3, 4)
    perm2 = [0, 2, 1, 3]
    result2 = input2.permute(*perm2)
    
    print(f"输入形状: {input2.shape}")
    print(f"转置轴: {perm2}")
    print(f"结果形状: {result2.shape}")
    print(f"输入范围: [{input2.min():.6f}, {input2.max():.6f}]")
    print(f"结果范围: [{result2.min():.6f}, {result2.max():.6f}]")
    
    save_4d_tensor(input2, "transpose_case2_input.npy")
    save_4d_tensor(result2, "transpose_case2_result.npy")
    
    perm_params2 = np.array(perm2)
    np.save("transpose_case2_params.npy", perm_params2)
    print(f"保存参数: {perm2}")
    print("保存到: transpose_case2_*.npy\n")
    
    # 测试案例3: 大尺寸张量
    print("生成测试案例3: 大尺寸张量")
    input3 = torch.randn(4, 8, 6, 7)
    perm3 = [0, 2, 1, 3]
    result3 = input3.permute(*perm3)
    
    print(f"输入形状: {input3.shape}")
    print(f"转置轴: {perm3}")
    print(f"结果形状: {result3.shape}")
    print(f"输入范围: [{input3.min():.6f}, {input3.max():.6f}]")
    print(f"结果范围: [{result3.min():.6f}, {result3.max():.6f}]")
    
    save_4d_tensor(input3, "transpose_case3_input.npy")
    save_4d_tensor(result3, "transpose_case3_result.npy")
    
    perm_params3 = np.array(perm3)
    np.save("transpose_case3_params.npy", perm_params3)
    print(f"保存参数: {perm3}")
    print("保存到: transpose_case3_*.npy\n")
    
    # 测试案例4: 正方形维度
    print("生成测试案例4: 正方形维度")
    input4 = torch.randn(2, 4, 4, 3)
    perm4 = [0, 2, 1, 3]
    result4 = input4.permute(*perm4)
    
    print(f"输入形状: {input4.shape}")
    print(f"转置轴: {perm4}")
    print(f"结果形状: {result4.shape}")
    print(f"输入范围: [{input4.min():.6f}, {input4.max():.6f}]")
    print(f"结果范围: [{result4.min():.6f}, {result4.max():.6f}]")
    
    save_4d_tensor(input4, "transpose_case4_input.npy")
    save_4d_tensor(result4, "transpose_case4_result.npy")
    
    perm_params4 = np.array(perm4)
    np.save("transpose_case4_params.npy", perm_params4)
    print(f"保存参数: {perm4}")
    print("保存到: transpose_case4_*.npy\n")
    
    # 测试案例5: 边界测试
    print("生成测试案例5: 边界测试")
    input5 = torch.randn(1, 1, 2, 3)
    perm5 = [0, 2, 1, 3]
    result5 = input5.permute(*perm5)
    
    print(f"输入形状: {input5.shape}")
    print(f"转置轴: {perm5}")
    print(f"结果形状: {result5.shape}")
    print(f"输入范围: [{input5.min():.6f}, {input5.max():.6f}]")
    print(f"结果范围: [{result5.min():.6f}, {result5.max():.6f}]")
    
    save_4d_tensor(input5, "transpose_case5_input.npy")
    save_4d_tensor(result5, "transpose_case5_result.npy")
    
    perm_params5 = np.array(perm5)
    np.save("transpose_case5_params.npy", perm_params5)
    print(f"保存参数: {perm5}")
    print("保存到: transpose_case5_*.npy\n")
    
    print("=== Transpose 测试数据生成完成 ===")

if __name__ == "__main__":
    test_transpose()
