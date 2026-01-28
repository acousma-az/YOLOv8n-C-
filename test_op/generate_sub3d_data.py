import torch
import numpy as np

def save_3d_tensor(tensor, filename):
    """保存3D张量到npy文件"""
    numpy_data = tensor.detach().cpu().numpy()
    np.save(filename, numpy_data)
    print(f"保存 {filename}: {tensor.shape}")

def test_sub_3d():
    """生成sub_3d测试数据"""
    print("=== 生成 Sub_3D 对比测试数据 ===\n")
    
    # 测试案例1: 相同形状的减法
    print("生成测试案例1: 相同形状的减法")
    input1_1 = torch.randn(4, 3, 5)
    input2_1 = torch.randn(4, 3, 5)
    result1 = input1_1 - input2_1
    
    print(f"输入1形状: {input1_1.shape}")
    print(f"输入2形状: {input2_1.shape}")
    print(f"结果形状: {result1.shape}")
    print(f"输入1范围: [{input1_1.min():.6f}, {input1_1.max():.6f}]")
    print(f"输入2范围: [{input2_1.min():.6f}, {input2_1.max():.6f}]")
    print(f"结果范围: [{result1.min():.6f}, {result1.max():.6f}]")
    
    save_3d_tensor(input1_1, "sub3d_case1_input1.npy")
    save_3d_tensor(input2_1, "sub3d_case1_input2.npy")
    save_3d_tensor(result1, "sub3d_case1_result.npy")
    print("保存到: sub3d_case1_*.npy\n")
    
    # 测试案例2: 深度维度广播 (4,3,5) - (1,3,5)
    print("生成测试案例2: 深度维度广播")
    input1_2 = torch.randn(4, 3, 5)
    input2_2 = torch.randn(1, 3, 5)
    result2 = input1_2 - input2_2
    
    print(f"输入1形状: {input1_2.shape}")
    print(f"输入2形状: {input2_2.shape}")
    print(f"结果形状: {result2.shape}")
    print(f"输入1范围: [{input1_2.min():.6f}, {input1_2.max():.6f}]")
    print(f"输入2范围: [{input2_2.min():.6f}, {input2_2.max():.6f}]")
    print(f"结果范围: [{result2.min():.6f}, {result2.max():.6f}]")
    
    save_3d_tensor(input1_2, "sub3d_case2_input1.npy")
    save_3d_tensor(input2_2, "sub3d_case2_input2.npy")
    save_3d_tensor(result2, "sub3d_case2_result.npy")
    print("保存到: sub3d_case2_*.npy\n")
    
    # 测试案例3: 高度维度广播 (3,4,6) - (3,1,6)
    print("生成测试案例3: 高度维度广播")
    input1_3 = torch.randn(3, 4, 6)
    input2_3 = torch.randn(3, 1, 6)
    result3 = input1_3 - input2_3
    
    print(f"输入1形状: {input1_3.shape}")
    print(f"输入2形状: {input2_3.shape}")
    print(f"结果形状: {result3.shape}")
    print(f"输入1范围: [{input1_3.min():.6f}, {input1_3.max():.6f}]")
    print(f"输入2范围: [{input2_3.min():.6f}, {input2_3.max():.6f}]")
    print(f"结果范围: [{result3.min():.6f}, {result3.max():.6f}]")
    
    save_3d_tensor(input1_3, "sub3d_case3_input1.npy")
    save_3d_tensor(input2_3, "sub3d_case3_input2.npy")
    save_3d_tensor(result3, "sub3d_case3_result.npy")
    print("保存到: sub3d_case3_*.npy\n")
    
    # 测试案例4: 宽度维度广播 (2,3,5) - (2,3,1)
    print("生成测试案例4: 宽度维度广播")
    input1_4 = torch.randn(2, 3, 5)
    input2_4 = torch.randn(2, 3, 1)
    result4 = input1_4 - input2_4
    
    print(f"输入1形状: {input1_4.shape}")
    print(f"输入2形状: {input2_4.shape}")
    print(f"结果形状: {result4.shape}")
    print(f"输入1范围: [{input1_4.min():.6f}, {input1_4.max():.6f}]")
    print(f"输入2范围: [{input2_4.min():.6f}, {input2_4.max():.6f}]")
    print(f"结果范围: [{result4.min():.6f}, {result4.max():.6f}]")
    
    save_3d_tensor(input1_4, "sub3d_case4_input1.npy")
    save_3d_tensor(input2_4, "sub3d_case4_input2.npy")
    save_3d_tensor(result4, "sub3d_case4_result.npy")
    print("保存到: sub3d_case4_*.npy\n")
    
    # 测试案例5: 多维度广播 (4,3,5) - (1,1,1)
    print("生成测试案例5: 多维度广播")
    input1_5 = torch.randn(4, 3, 5)
    input2_5 = torch.randn(1, 1, 1)
    result5 = input1_5 - input2_5
    
    print(f"输入1形状: {input1_5.shape}")
    print(f"输入2形状: {input2_5.shape}")
    print(f"结果形状: {result5.shape}")
    print(f"输入1范围: [{input1_5.min():.6f}, {input1_5.max():.6f}]")
    print(f"输入2范围: [{input2_5.min():.6f}, {input2_5.max():.6f}]")
    print(f"结果范围: [{result5.min():.6f}, {result5.max():.6f}]")
    
    save_3d_tensor(input1_5, "sub3d_case5_input1.npy")
    save_3d_tensor(input2_5, "sub3d_case5_input2.npy")
    save_3d_tensor(result5, "sub3d_case5_result.npy")
    print("保存到: sub3d_case5_*.npy\n")
    
    print("=== Sub_3D 测试数据生成完成 ===")

if __name__ == "__main__":
    test_sub_3d()
