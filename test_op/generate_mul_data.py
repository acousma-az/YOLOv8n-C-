#!/usr/bin/env python3
import torch
import numpy as np

def generate_mul_test_data():
    """生成mul算子的PyTorch对比测试数据"""
    print("=== 生成 Mul 对比测试数据 ===\n")
    
    # 设置随机种子以获得可重复的结果
    torch.manual_seed(42)
    
    # 测试案例1: 相同形状的4D张量乘法
    print("生成测试案例1: 相同形状4D张量乘法")
    input1_case1 = torch.randn(2, 4, 8, 8)
    input2_case1 = torch.randn(2, 4, 8, 8)
    
    # PyTorch乘法计算
    result1 = input1_case1 * input2_case1
    
    print(f"输入1形状: {input1_case1.shape}")
    print(f"输入2形状: {input2_case1.shape}")
    print(f"结果形状: {result1.shape}")
    
    # 保存数据
    np.save('mul_case1_input1.npy', input1_case1.numpy())
    np.save('mul_case1_input2.npy', input2_case1.numpy())
    np.save('mul_case1_result.npy', result1.numpy())
    print("保存到: mul_case1_*.npy\n")
    
    # 测试案例2: 通道维度广播
    print("生成测试案例2: 通道维度广播")
    input1_case2 = torch.randn(1, 8, 16, 16)
    input2_case2 = torch.randn(1, 1, 16, 16)  # 通道维度为1，会广播
    
    result2 = input1_case2 * input2_case2
    
    print(f"输入1形状: {input1_case2.shape}")
    print(f"输入2形状: {input2_case2.shape}")
    print(f"结果形状: {result2.shape}")
    
    np.save('mul_case2_input1.npy', input1_case2.numpy())
    np.save('mul_case2_input2.npy', input2_case2.numpy())
    np.save('mul_case2_result.npy', result2.numpy())
    print("保存到: mul_case2_*.npy\n")
    
    # 测试案例3: 多维度广播
    print("生成测试案例3: 多维度广播")
    input1_case3 = torch.randn(2, 6, 12, 12)
    input2_case3 = torch.randn(1, 1, 1, 12)  # 多个维度广播
    
    result3 = input1_case3 * input2_case3
    
    print(f"输入1形状: {input1_case3.shape}")
    print(f"输入2形状: {input2_case3.shape}")
    print(f"结果形状: {result3.shape}")
    
    np.save('mul_case3_input1.npy', input1_case3.numpy())
    np.save('mul_case3_input2.npy', input2_case3.numpy())
    np.save('mul_case3_result.npy', result3.numpy())
    print("保存到: mul_case3_*.npy\n")
    
    # 测试案例4: Batch维度广播
    print("生成测试案例4: Batch维度广播")
    input1_case4 = torch.randn(4, 16, 8, 8)
    input2_case4 = torch.randn(1, 16, 8, 8)  # Batch维度为1，会广播
    
    result4 = input1_case4 * input2_case4
    
    print(f"输入1形状: {input1_case4.shape}")
    print(f"输入2形状: {input2_case4.shape}")
    print(f"结果形状: {result4.shape}")
    
    np.save('mul_case4_input1.npy', input1_case4.numpy())
    np.save('mul_case4_input2.npy', input2_case4.numpy())
    np.save('mul_case4_result.npy', result4.numpy())
    print("保存到: mul_case4_*.npy\n")
    
    print("=== Mul 测试数据生成完成 ===")

if __name__ == "__main__":
    generate_mul_test_data()
