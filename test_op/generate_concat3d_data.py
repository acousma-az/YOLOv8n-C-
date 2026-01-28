#!/usr/bin/env python3
import torch
import numpy as np

def generate_concat3d_test_data():
    """生成concat_3d的PyTorch对比测试数据"""
    print("=== 生成 Concat_3D 对比测试数据 ===\n")
    
    # 测试案例1: 沿axis=1连接
    print("生成测试案例: 沿axis=1连接")
    input1_case1 = torch.randn(4, 3, 5)
    input2_case1 = torch.randn(4, 2, 5)
    input3_case1 = torch.randn(4, 4, 5)
    result_case1 = torch.cat([input1_case1, input2_case1, input3_case1], dim=1)
    
    print(f"输入1形状: {input1_case1.shape}")
    print(f"输入2形状: {input2_case1.shape}")
    print(f"输入3形状: {input3_case1.shape}")
    print(f"结果形状: {result_case1.shape}")
    
    # 保存数据
    np.save('concat3d_case1_axis1_input1.npy', input1_case1.numpy())
    np.save('concat3d_case1_axis1_input2.npy', input2_case1.numpy())
    np.save('concat3d_case1_axis1_input3.npy', input3_case1.numpy())
    np.save('concat3d_case1_axis1_result.npy', result_case1.numpy())
    print("保存到: concat3d_case1_axis1_*.npy\n")
    
    # 测试案例2: 沿axis=2连接
    print("生成测试案例: 沿axis=2连接")
    input1_case2 = torch.randn(3, 4, 2)
    input2_case2 = torch.randn(3, 4, 3)
    input3_case2 = torch.randn(3, 4, 1)
    result_case2 = torch.cat([input1_case2, input2_case2, input3_case2], dim=2)
    
    print(f"输入1形状: {input1_case2.shape}")
    print(f"输入2形状: {input2_case2.shape}")
    print(f"输入3形状: {input3_case2.shape}")
    print(f"结果形状: {result_case2.shape}")
    
    # 保存数据
    np.save('concat3d_case2_axis2_input1.npy', input1_case2.numpy())
    np.save('concat3d_case2_axis2_input2.npy', input2_case2.numpy())
    np.save('concat3d_case2_axis2_input3.npy', input3_case2.numpy())
    np.save('concat3d_case2_axis2_result.npy', result_case2.numpy())
    print("保存到: concat3d_case2_axis2_*.npy\n")
    
    # 测试案例3: 沿axis=0连接
    print("生成测试案例: 沿axis=0连接")
    input1_case3 = torch.randn(2, 6, 8)
    input2_case3 = torch.randn(3, 6, 8)
    result_case3 = torch.cat([input1_case3, input2_case3], dim=0)
    
    print(f"输入1形状: {input1_case3.shape}")
    print(f"输入2形状: {input2_case3.shape}")
    print(f"结果形状: {result_case3.shape}")
    
    # 保存数据
    np.save('concat3d_case3_axis0_input1.npy', input1_case3.numpy())
    np.save('concat3d_case3_axis0_input2.npy', input2_case3.numpy())
    np.save('concat3d_case3_axis0_result.npy', result_case3.numpy())
    print("保存到: concat3d_case3_axis0_*.npy\n")
    
    print("=== Concat_3D 测试数据生成完成 ===")

if __name__ == "__main__":
    generate_concat3d_test_data()
