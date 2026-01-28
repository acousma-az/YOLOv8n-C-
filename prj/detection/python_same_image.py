#!/usr/bin/env python3
"""
使用相同图像进行直接对比
"""

import numpy as np
from PIL import Image
import os

def python_preprocessing_same_image():
    """
    使用与C++相同的图像进行预处理
    """
    image_path = "/home/tpu1/project/heyang/yolo/end_640x640.jpg"
    
    print(f"Python预处理同一图像: {image_path}")
    print("=" * 50)
    
    # 加载图像
    img = Image.open(image_path)
    print(f"成功加载图像: {image_path}")
    print(f"PIL加载的图像模式: {img.mode}")
    print(f"原始尺寸: {img.size[0]}x{img.size[1]}")
    
    # 确保是RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    else:
        print("图像已经是RGB格式")
    
    # 转换为numpy数组
    img_array = np.array(img, dtype=np.uint8)
    print(f"原始数组形状: {img_array.shape}")
    print(f"原始RGB (0,0): {img_array[0, 0]}")
    
    # 归一化
    normalized = img_array.astype(np.float32) / 255.0
    print(f"归一化RGB (0,0): {normalized[0, 0]}")
    
    # HWC -> CHW
    chw_python = np.transpose(normalized, (2, 0, 1))
    print(f"CHW形状: {chw_python.shape}")
    print(f"CHW格式 (0,0): R={chw_python[0,0,0]}, G={chw_python[1,0,0]}, B={chw_python[2,0,0]}")
    
    # 统计信息
    print(f"\nPython结果统计:")
    print(f"  - 数据范围: [{chw_python.min():.6f}, {chw_python.max():.6f}]")
    print(f"  - 平均值: {chw_python.mean():.6f}")
    print(f"  - 总元素数: {chw_python.size}")
    
    # 各通道统计
    print(f"\n各通道统计信息:")
    for c in range(3):
        channel_data = chw_python[c]
        print(f"  通道 {c}: 均值={channel_data.mean():.6f}, "
              f"范围=[{channel_data.min():.6f}, {channel_data.max():.6f}]")
    
    # 保存前10个值用于精确对比
    with open("python_preprocessing_result.txt", 'w') as f:
        f.write("# Python 预处理结果\n")
        f.write(f"# Shape: [{chw_python.shape[0]}, {chw_python.shape[1]}, {chw_python.shape[2]}]\n")
        f.write("# 前10个像素值 (CHW格式):\n")
        flat_data = chw_python.flatten()
        for i in range(min(10, len(flat_data))):
            f.write(f"{flat_data[i]}\n")
    
    print("Python预处理结果已保存到: python_preprocessing_result.txt")
    
    return chw_python

def compare_results():
    """
    对比C++和Python的结果
    """
    print("\n\n结果对比分析")
    print("=" * 50)
    
    # 读取C++结果
    cpp_file = "/home/tpu1/project/heyang/yolo/detection/cpp_preprocessing_result.txt"
    python_file = "python_preprocessing_result.txt"
    
    if os.path.exists(cpp_file) and os.path.exists(python_file):
        print("读取前10个像素值进行精确对比...")
        
        # 读取C++结果
        cpp_values = []
        with open(cpp_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        cpp_values.append(float(line))
                    except:
                        pass
        
        # 读取Python结果
        python_values = []
        with open(python_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        python_values.append(float(line))
                    except:
                        pass
        
        print(f"C++前10个值: {cpp_values}")
        print(f"Python前10个值: {python_values}")
        
        # 计算差异
        if len(cpp_values) >= 10 and len(python_values) >= 10:
            differences = []
            for i in range(10):
                diff = abs(cpp_values[i] - python_values[i])
                differences.append(diff)
                print(f"像素 {i}: C++={cpp_values[i]:.6f}, Python={python_values[i]:.6f}, 差异={diff:.2e}")
            
            max_diff = max(differences)
            avg_diff = sum(differences) / len(differences)
            
            print(f"\n差异统计:")
            print(f"最大差异: {max_diff:.2e}")
            print(f"平均差异: {avg_diff:.2e}")
            
            if max_diff < 1e-6:
                print("✅ 结果几乎完全一致 (差异 < 1e-6)")
            elif max_diff < 1e-4:
                print("✅ 结果高度一致 (差异 < 1e-4)")
            elif max_diff < 1e-2:
                print("⚠️  结果基本一致 (差异 < 1e-2)")
            else:
                print("❌ 结果存在显著差异")
    
    print("\n总结:")
    print("从控制台输出对比:")
    print("C++  (0,0)像素: R=0.670588, G=0.603922, B=0.533333")
    print("Python (预期): 需要运行相同图像...")

if __name__ == "__main__":
    result = python_preprocessing_same_image()
    compare_results()
