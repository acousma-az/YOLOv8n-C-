#!/usr/bin/env python3
import torch
import numpy as np

def generate_div3d_test_data():
    """生成div_3d算子的PyTorch对比测试数据"""
    print("=== 生成 Div_3D 对比测试数据 ===\n")
    
    # 设置随机种子以获得可重复的结果
    torch.manual_seed(42)
    
    # 测试案例1: 基本3D张量除法
    print("生成测试案例1: 基本3D张量除法")
    input1 = torch.randn(8, 12, 16)
    divisor1 = 2.5
    
    # PyTorch除法计算
    result1 = input1 / divisor1
    
    print(f"输入形状: {input1.shape}")
    print(f"除数: {divisor1}")
    print(f"结果形状: {result1.shape}")
    
    # 保存数据
    np.save('div3d_case1_input.npy', input1.numpy())
    np.save('div3d_case1_result.npy', result1.numpy())
    # 保存除数（作为单个元素的numpy数组）
    np.save('div3d_case1_divisor.npy', np.array([divisor1], dtype=np.float32))
    print("保存到: div3d_case1_*.npy\n")
    
    # 测试案例2: 小数除法
    print("生成测试案例2: 小数除法")
    input2 = torch.randn(4, 6, 8) * 10  # 放大数值
    divisor2 = 0.125  # 1/8
    
    result2 = input2 / divisor2
    
    print(f"输入形状: {input2.shape}")
    print(f"除数: {divisor2}")
    print(f"结果形状: {result2.shape}")
    
    np.save('div3d_case2_input.npy', input2.numpy())
    np.save('div3d_case2_result.npy', result2.numpy())
    np.save('div3d_case2_divisor.npy', np.array([divisor2], dtype=np.float32))
    print("保存到: div3d_case2_*.npy\n")
    
    # 测试案例3: 负数除法
    print("生成测试案例3: 负数除法")
    input3 = torch.randn(6, 8, 10)
    divisor3 = -3.0
    
    result3 = input3 / divisor3
    
    print(f"输入形状: {input3.shape}")
    print(f"除数: {divisor3}")
    print(f"结果形状: {result3.shape}")
    
    np.save('div3d_case3_input.npy', input3.numpy())
    np.save('div3d_case3_result.npy', result3.numpy())
    np.save('div3d_case3_divisor.npy', np.array([divisor3], dtype=np.float32))
    print("保存到: div3d_case3_*.npy\n")
    
    # 测试案例4: 大数除法
    print("生成测试案例4: 大数除法")
    input4 = torch.randn(2, 4, 6) * 1000  # 大数值
    divisor4 = 1000.0
    
    result4 = input4 / divisor4
    
    print(f"输入形状: {input4.shape}")
    print(f"除数: {divisor4}")
    print(f"结果形状: {result4.shape}")
    
    np.save('div3d_case4_input.npy', input4.numpy())
    np.save('div3d_case4_result.npy', result4.numpy())
    np.save('div3d_case4_divisor.npy', np.array([divisor4], dtype=np.float32))
    print("保存到: div3d_case4_*.npy\n")
    
    print("=== Div_3D 测试数据生成完成 ===")

if __name__ == "__main__":
    generate_div3d_test_data()
