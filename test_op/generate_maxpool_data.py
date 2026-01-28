#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import numpy as np

def generate_maxpool_test_data():
    """生成maxpool算子的PyTorch对比测试数据"""
    print("=== 生成 MaxPool 对比测试数据 ===\n")
    
    # 设置随机种子以获得可重复的结果
    torch.manual_seed(42)
    
    # 测试案例1: 基本2x2 maxpool
    print("生成测试案例1: 基本2x2 MaxPool")
    input1 = torch.randn(1, 3, 8, 8)
    kernel_size1, stride1, padding1 = 2, 2, 0
    
    # PyTorch maxpool计算
    result1 = F.max_pool2d(input1, kernel_size=kernel_size1, stride=stride1, padding=padding1)
    
    print(f"输入形状: {input1.shape}")
    print(f"结果形状: {result1.shape}")
    print(f"参数: kernel_size={kernel_size1}, stride={stride1}, padding={padding1}")
    
    # 保存数据
    np.save('maxpool_case1_input.npy', input1.numpy())
    np.save('maxpool_case1_result.npy', result1.numpy())
    print("保存到: maxpool_case1_*.npy\n")
    
    # 测试案例2: 3x3 maxpool with padding
    print("生成测试案例2: 3x3 MaxPool with padding")
    input2 = torch.randn(2, 16, 16, 16)
    kernel_size2, stride2, padding2 = 3, 2, 1
    
    result2 = F.max_pool2d(input2, kernel_size=kernel_size2, stride=stride2, padding=padding2)
    
    print(f"输入形状: {input2.shape}")
    print(f"结果形状: {result2.shape}")
    print(f"参数: kernel_size={kernel_size2}, stride={stride2}, padding={padding2}")
    
    np.save('maxpool_case2_input.npy', input2.numpy())
    np.save('maxpool_case2_result.npy', result2.numpy())
    print("保存到: maxpool_case2_*.npy\n")
    
    # 测试案例3: 2x2 maxpool with stride=1 (重叠池化)
    print("生成测试案例3: 2x2 MaxPool with stride=1")
    input3 = torch.randn(1, 8, 6, 6)
    kernel_size3, stride3, padding3 = 2, 1, 0
    
    result3 = F.max_pool2d(input3, kernel_size=kernel_size3, stride=stride3, padding=padding3)
    
    print(f"输入形状: {input3.shape}")
    print(f"结果形状: {result3.shape}")
    print(f"参数: kernel_size={kernel_size3}, stride={stride3}, padding={padding3}")
    
    np.save('maxpool_case3_input.npy', input3.numpy())
    np.save('maxpool_case3_result.npy', result3.numpy())
    print("保存到: maxpool_case3_*.npy\n")
    
    # 测试案例4: 大尺寸输入
    print("生成测试案例4: 大尺寸输入")
    input4 = torch.randn(2, 32, 32, 32)
    kernel_size4, stride4, padding4 = 2, 2, 0
    
    result4 = F.max_pool2d(input4, kernel_size=kernel_size4, stride=stride4, padding=padding4)
    
    print(f"输入形状: {input4.shape}")
    print(f"结果形状: {result4.shape}")
    print(f"参数: kernel_size={kernel_size4}, stride={stride4}, padding={padding4}")
    
    np.save('maxpool_case4_input.npy', input4.numpy())
    np.save('maxpool_case4_result.npy', result4.numpy())
    print("保存到: maxpool_case4_*.npy\n")
    
    print("=== MaxPool 测试数据生成完成 ===")

if __name__ == "__main__":
    generate_maxpool_test_data()
