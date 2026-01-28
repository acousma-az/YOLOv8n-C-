#!/usr/bin/env python3
"""
生成测试数据供C++和Python对比使用
"""

import torch
import numpy as np
import os

def generate_test_data():
    """生成固定的测试数据"""
    print("=== 生成对比测试数据 ===")
    
    # 使用固定种子确保可重复
    torch.manual_seed(12345)
    np.random.seed(12345)
    
    # 测试案例1: 基本连接
    print("生成测试案例1: 基本连接")
    tensor1 = torch.randn(2, 3, 4, 4)
    tensor2 = torch.randn(2, 2, 4, 4)
    tensor3 = torch.randn(2, 1, 4, 4)
    
    # PyTorch计算结果
    result_pytorch = torch.cat([tensor1, tensor2, tensor3], dim=1)
    
    # 保存数据
    np.save('test1_input1.npy', tensor1.numpy())
    np.save('test1_input2.npy', tensor2.numpy())
    np.save('test1_input3.npy', tensor3.numpy())
    np.save('test1_result_pytorch.npy', result_pytorch.numpy())
    
    print(f"输入1形状: {tensor1.shape}")
    print(f"输入2形状: {tensor2.shape}")
    print(f"输入3形状: {tensor3.shape}")
    print(f"PyTorch结果形状: {result_pytorch.shape}")
    print(f"保存到: test1_*.npy")
    print()
    
    # 测试案例2: 不同通道数
    print("生成测试案例2: 不同通道数")
    tensor4 = torch.randn(1, 8, 6, 6)
    tensor5 = torch.randn(1, 4, 6, 6)
    
    result_pytorch2 = torch.cat([tensor4, tensor5], dim=1)
    
    np.save('test2_input1.npy', tensor4.numpy())
    np.save('test2_input2.npy', tensor5.numpy())
    np.save('test2_result_pytorch.npy', result_pytorch2.numpy())
    
    print(f"输入1形状: {tensor4.shape}")
    print(f"输入2形状: {tensor5.shape}")
    print(f"PyTorch结果形状: {result_pytorch2.shape}")
    print(f"保存到: test2_*.npy")
    print()
    
    # 显示一些数值用于调试
    print("测试案例1 - 部分数值:")
    print(f"输入1[0,0,0,:3]: {tensor1[0,0,0,:3].numpy()}")
    print(f"输入2[0,0,0,:3]: {tensor2[0,0,0,:3].numpy()}")
    print(f"结果[0,0,0,:3]: {result_pytorch[0,0,0,:3].numpy()}")
    print(f"结果[0,3,0,:3]: {result_pytorch[0,3,0,:3].numpy()}")  # 第二个tensor的第一个通道
    print()

if __name__ == "__main__":
    os.chdir('/home/tpu1/project/heyang/test_operator')
    generate_test_data()
    print("数据生成完成！")
