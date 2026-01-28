#ifndef IMAGE_RESIZE_H
#define IMAGE_RESIZE_H

/**
 * @brief 图像缩放并归一化到 CHW 格式
 * 
 * @param image_data 原始图像数据（HWC，RGB，unsigned char*）
 * @param orig_w 原始宽度
 * @param orig_h 原始高度
 * @param target_w 目标宽度
 * @param target_h 目标高度
 * @return float* 返回归一化后的 float 数组，CHW 格式，需手动 free
 */
float* resize_and_normalize(unsigned char* image_data, int orig_w, int orig_h, int target_w, int target_h);

#endif // IMAGE_RESIZE_H
