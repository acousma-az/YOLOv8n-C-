#!/usr/bin/env python3
"""
生成 add_4d 测试数据，用于 C++ 和 PyTorch 直接对比
"""

import torch
import numpy as np
import os

def generate_add4d_test_data():
    """生成add_4d测试数据"""
    print("=== 生成 Add_4D 对比测试数据 ===\n")
    
    # 设置固定随机种子
    torch.manual_seed(12345)
    np.random.seed(12345)
    
    test_cases = [
        {
            "name": "case1_same_shape",
            "shape1": (2, 4, 8, 8),
            "shape2": (2, 4, 8, 8),
            "desc": "相同形状张量"
        },
        {
            "name": "case2_batch_broadcast", 
            "shape1": (4, 8, 16, 16),
            "shape2": (1, 8, 16, 16),
            "desc": "Batch维度广播"
        },
        {
            "name": "case3_multi_broadcast",
            "shape1": (2, 8, 12, 12),
            "shape2": (1, 1, 1, 12),
            "desc": "多维度广播"
        }
    ]
    
    for case in test_cases:
        print(f"生成测试案例: {case['desc']}")
        print(f"形状1: {case['shape1']}, 形状2: {case['shape2']}")
        
        # 生成输入张量
        tensor1 = torch.randn(case['shape1']) * 5
        tensor2 = torch.randn(case['shape2']) * 5
        
        # PyTorch计算结果
        result_pytorch = tensor1 + tensor2
        
        # 保存数据
        np.save(f"add4d_{case['name']}_input1.npy", tensor1.numpy())
        np.save(f"add4d_{case['name']}_input2.npy", tensor2.numpy())
        np.save(f"add4d_{case['name']}_result.npy", result_pytorch.numpy())
        
        print(f"输入1形状: {tensor1.shape}")
        print(f"输入2形状: {tensor2.shape}")
        print(f"结果形状: {result_pytorch.shape}")
        print(f"保存到: add4d_{case['name']}_*.npy")
        print()

if __name__ == "__main__":
    generate_add4d_test_data()
    print("=== Add_4D 测试数据生成完成 ===")
