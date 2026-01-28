#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import numpy as np

def generate_conv_test_data():
    """生成conv算子的PyTorch对比测试数据"""
    print("=== 生成 Conv 对比测试数据 ===\n")
    
    # 设置随机种子以获得可重复的结果
    torch.manual_seed(42)
    
    # 测试案例1: 基本3x3卷积
    print("生成测试案例1: 基本3x3卷积")
    batch, in_channels, in_height, in_width = 1, 3, 8, 8
    out_channels, kernel_size = 16, 3
    stride, padding = 1, 1
    
    input1 = torch.randn(batch, in_channels, in_height, in_width)
    weight1 = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
    bias1 = torch.randn(out_channels)
    
    # PyTorch卷积计算
    result1 = F.conv2d(input1, weight1, bias1, stride=stride, padding=padding)
    
    print(f"输入形状: {input1.shape}")
    print(f"权重形状: {weight1.shape}")
    print(f"偏置形状: {bias1.shape}")
    print(f"结果形状: {result1.shape}")
    print(f"参数: stride={stride}, padding={padding}")
    
    # 保存数据
    np.save('conv_case1_input.npy', input1.numpy())
    np.save('conv_case1_weight.npy', weight1.numpy())
    np.save('conv_case1_bias.npy', bias1.numpy())
    np.save('conv_case1_result.npy', result1.numpy())
    print("保存到: conv_case1_*.npy\n")
    
    # 测试案例2: 1x1卷积 (pointwise)
    print("生成测试案例2: 1x1卷积")
    batch, in_channels, in_height, in_width = 2, 32, 16, 16
    out_channels, kernel_size = 64, 1
    stride, padding = 1, 0
    
    input2 = torch.randn(batch, in_channels, in_height, in_width)
    weight2 = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
    bias2 = torch.randn(out_channels)
    
    result2 = F.conv2d(input2, weight2, bias2, stride=stride, padding=padding)
    
    print(f"输入形状: {input2.shape}")
    print(f"权重形状: {weight2.shape}")
    print(f"偏置形状: {bias2.shape}")
    print(f"结果形状: {result2.shape}")
    print(f"参数: stride={stride}, padding={padding}")
    
    np.save('conv_case2_input.npy', input2.numpy())
    np.save('conv_case2_weight.npy', weight2.numpy())
    np.save('conv_case2_bias.npy', bias2.numpy())
    np.save('conv_case2_result.npy', result2.numpy())
    print("保存到: conv_case2_*.npy\n")
    
    # 测试案例3: stride=2的卷积
    print("生成测试案例3: stride=2卷积")
    batch, in_channels, in_height, in_width = 1, 8, 32, 32
    out_channels, kernel_size = 16, 3
    stride, padding = 2, 1
    
    input3 = torch.randn(batch, in_channels, in_height, in_width)
    weight3 = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
    bias3 = torch.randn(out_channels)
    
    result3 = F.conv2d(input3, weight3, bias3, stride=stride, padding=padding)
    
    print(f"输入形状: {input3.shape}")
    print(f"权重形状: {weight3.shape}")
    print(f"偏置形状: {bias3.shape}")
    print(f"结果形状: {result3.shape}")
    print(f"参数: stride={stride}, padding={padding}")
    
    np.save('conv_case3_input.npy', input3.numpy())
    np.save('conv_case3_weight.npy', weight3.numpy())
    np.save('conv_case3_bias.npy', bias3.numpy())
    np.save('conv_case3_result.npy', result3.numpy())
    print("保存到: conv_case3_*.npy\n")
    
    print("=== Conv 测试数据生成完成 ===")

if __name__ == "__main__":
    generate_conv_test_data()
