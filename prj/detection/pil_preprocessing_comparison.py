#!/usr/bin/env python3
"""
PIL图像预处理对比脚本
对比PIL和C++ stb_image的图像预处理步骤
"""

import numpy as np
from PIL import Image
import os

def pil_preprocessing(image_path):
    """
    使用PIL进行图像预处理，模拟C++代码的处理步骤
    """
    print(f"使用PIL加载图像: {image_path}")
    
    # 步骤1: 使用PIL加载图像
    try:
        img = Image.open(image_path)
        print(f"成功加载图像: {image_path}")
        print(f"PIL加载的图像模式: {img.mode}")
        print(f"原始尺寸: {img.size[0]}x{img.size[1]}")  # PIL返回(width, height)
        
        # 确保图像是RGB格式
        if img.mode != 'RGB':
            print(f"转换图像模式从 {img.mode} 到 RGB")
            img = img.convert('RGB')
        else:
            print("图像已经是RGB格式")
            
    except Exception as e:
        print(f"PIL加载图像失败: {e}")
        return None
    
    # 检查尺寸
    width, height = img.size
    if width == 640 and height == 640:
        print("图片尺寸已经是640x640，跳过resize处理")
    else:
        print(f"警告: 图片尺寸不是640x640，YOLOv8可能需要640x640输入")
    
    print("\n开始YOLO预处理步骤:")
    
    # 步骤1: 转换为numpy数组 (默认是HWC格式, RGB)
    print("步骤1: PIL转换为numpy数组 (HWC格式)")
    img_array = np.array(img, dtype=np.uint8)
    print(f"转换后shape: {img_array.shape} (HWC格式)")
    print(f"数据类型: {img_array.dtype}")
    
    # 打印原始RGB像素值 (0,0)
    print(f"  原始RGB (0,0): {img_array[0, 0]}")  # [R, G, B]
    
    # 步骤2: 归一化 (0-255 -> 0-1)
    print("步骤2: 像素值归一化 (0-255 -> 0-1)")
    normalized_data = img_array.astype(np.float32) / 255.0
    
    # 打印归一化后的像素值 (0,0)
    print(f"  归一化RGB (0,0): {normalized_data[0, 0]}")
    
    # 步骤3: HWC -> CHW 维度转换
    print("步骤3: HWC -> CHW 维度转换")
    print(f"转换前: {normalized_data.shape} (HWC)")
    
    # 使用numpy的transpose进行维度转换: (H,W,C) -> (C,H,W)
    chw_data = np.transpose(normalized_data, (2, 0, 1))
    print(f"转换后: {chw_data.shape} (CHW)")
    
    # 打印CHW转换后的像素值 (0,0)
    print(f"  CHW格式 (0,0): R={chw_data[0, 0, 0]}, G={chw_data[1, 0, 0]}, B={chw_data[2, 0, 0]}")
    
    # 打印统计信息
    print("预处理完成统计信息:")
    print(f"  - 数据范围: [{chw_data.min():.6f}, {chw_data.max():.6f}]")
    print(f"  - 平均值: {chw_data.mean():.6f}")
    print(f"  - 总元素数: {chw_data.size}")
    
    # 各通道统计信息
    print("\n各通道统计信息:")
    channel_names = ['R', 'G', 'B']
    for c in range(3):
        channel_data = chw_data[c]
        print(f"  通道 {c} ({channel_names[c]}): 均值={channel_data.mean():.6f}, "
              f"范围=[{channel_data.min():.6f}, {channel_data.max():.6f}]")
    
    return chw_data

def compare_with_cpp_output():
    """
    比较PIL和C++的预处理差异
    """
    print("\n" + "="*60)
    print("PIL vs C++ stb_image 预处理对比分析")
    print("="*60)
    
    print("\n1. 图像加载差异:")
    print("   PIL:")
    print("   - 使用PIL.Image.open()加载")
    print("   - 默认加载为RGB格式")
    print("   - 返回PIL Image对象")
    print("   - 坐标系: (width, height)")
    
    print("\n   C++ stb_image:")
    print("   - 使用stbi_load()加载")
    print("   - desired_channels=3强制加载为RGB")
    print("   - 返回unsigned char*指针")
    print("   - 坐标系: (width, height)")
    
    print("\n2. 数据格式差异:")
    print("   PIL:")
    print("   - np.array(img) -> uint8数组，HWC格式")
    print("   - 像素排列: [H, W, C]")
    print("   - 内存布局: 连续的HWC")
    
    print("\n   C++ stb_image:")
    print("   - stbi_load -> unsigned char*, HWC格式")
    print("   - 像素排列: [H, W, C]")
    print("   - 内存布局: 连续的HWC")
    
    print("\n3. 归一化方式:")
    print("   PIL: array.astype(np.float32) / 255.0")
    print("   C++: (float)pixel_value / 255.0f")
    print("   -> 两者完全一致")
    
    print("\n4. 维度转换:")
    print("   PIL: np.transpose(array, (2, 0, 1))  # HWC -> CHW")
    print("   C++: 手动循环转换 HWC -> CHW")
    print("   -> 结果应该一致，但实现方式不同")
    
    print("\n5. 潜在差异点:")
    print("   - 图像解码库差异: PIL vs stb_image")
    print("   - 浮点精度: numpy float32 vs C++ float")
    print("   - 内存对齐和缓存效应")
    print("   - 颜色空间解释差异（极少见）")

def main():
    # 查找图像文件
    image_path = "/home/tpu1/project/heyang/yolo/detection/end_640x640.jpg"
    
    if not os.path.exists(image_path):
        print(f"图像文件不存在: {image_path}")
        print("请确保文件路径正确")
        return
    
    # 执行PIL预处理
    result = pil_preprocessing(image_path)
    
    if result is not None:
        # 保存结果以便对比
        output_path = "/home/tpu1/project/heyang/yolo/detection/pil_preprocessed_output.npy"
        np.save(output_path, result)
        print(f"\nPIL预处理结果已保存到: {output_path}")
        print(f"可以使用np.load('{output_path}')加载数据进行对比")
    
    # 输出对比分析
    compare_with_cpp_output()

if __name__ == "__main__":
    main()
