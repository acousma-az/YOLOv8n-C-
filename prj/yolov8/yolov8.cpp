#include <iostream>
#include <vector>
#include "swish_block/swish_block0.h"
#include "swish_block/swish_block1.h"
#include "swish_block/swish_block2.h"
#include "swish_block/swish_block2_cv2.h"
#include "complex_block/complex_module0.h"
#include "swish_block/swish_block3.h"
#include "swish_block/swish_block4_cv1.h"
#include "swish_block/swish_block4_cv2.h"
#include "swish_block/swish_block5.h"
#include "swish_block/swish_block6_cv1.h"
#include "swish_block/swish_block6_cv2.h"
#include "swish_block/swish_block7.h"
#include "swish_block/swish_block8_cv1.h"
#include "swish_block/swish_block8_cv2.h"
#include "swish_block/swish_block9_cv1.h"
#include "complex_block/more_complex_module0.h"
#include "complex_block/more_complex_module1.h"
#include "complex_block/complex_module1.h"
#include "swish_block/swish_block9_cv2.h"
#include "swish_block/swish_block12_cv1.h"
#include "complex_block/simple_module0.h"
#include "swish_block/swish_block12_cv2.h"
#include "swish_block/swish_block15_cv1.h"
#include "complex_block/simple_module1.h"
#include "swish_block/swish_block15_cv2.h"
#include "swish_block/swish_block16.h"
#include "swish_block/swish_block22_cv2_0_0.h"
#include "swish_block/swish_block22_cv2_0_1.h"
#include "swish_block/swish_block22_cv3_0_0.h"
#include "swish_block/swish_block22_cv3_0_1.h"
#include "operator/conv.h"
#include "weight/weights.h"
#include "swish_block/swish_block18_cv1.h"
#include "complex_block/simple_module2.h"
#include "swish_block/swish_block18_cv2.h"
#include "swish_block/swish_block22_cv2_1_0.h"
#include "swish_block/swish_block22_cv2_1_1.h"
#include "swish_block/swish_block22_cv3_1_0.h"
#include "swish_block/swish_block22_cv3_1_1.h"
#include "swish_block/swish_block19.h"
#include "swish_block/swish_block21_cv1.h"
#include "complex_block/simple_module3.h"
#include "swish_block/swish_block21_cv2.h"
#include "swish_block/swish_block22_cv2_2_0.h"
#include "swish_block/swish_block22_cv2_2_1.h"
#include "swish_block/swish_block22_cv3_2_0.h"
#include "swish_block/swish_block22_cv3_2_1.h"
#include "operator/concat_3d.h"
#include "operator/split_3d.h"
#include "operator/reshape_4d_to_3d.h"
#include "operator/reshape_3d_to_4d.h"
#include "operator/maxpool.h"
#include "operator/concat.h"
#include "operator/resize.h"
#include "operator/transpose.h"
#include "operator/softmax.h"
#include "operator/slice.h"
#include "operator/sub_3d.h"
#include "operator/add_3d.h"
#include "operator/div_3d.h"
#include "operator/mul_3d.h"
#include "operator/sigmoid_3d.h"

#include "yolov8.h"

/*
std::vector<std::vector<std::vector<std::vector<float>>>> yolov8(float* input_chw) {

    std::vector<std::vector<std::vector<std::vector<float>>>> input_image(
        1, std::vector<std::vector<std::vector<float>>>(
            3, std::vector<std::vector<float>>(
                640, std::vector<float>(640, 0.0f)
            )
        )
    );
*/
std::vector<std::vector<std::vector<float>>> yolov8(float* input_chw) {

    std::vector<std::vector<std::vector<std::vector<float>>>> input_image(
        1, std::vector<std::vector<std::vector<float>>>(
            3, std::vector<std::vector<float>>(
                640, std::vector<float>(640, 0.0f)
            )
        )
    );


    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < 640; ++h) {
            for (int w = 0; w < 640; ++w) {
                int idx = c * 640 * 640 + h * 640 + w;
                input_image[0][c][h][w] = input_chw[idx];
            }
        }
    }

    std::cout << "Input image size: " << input_image.size() << "*" 
              << input_image[0].size() << "*" 
              << input_image[0][0].size() << "*" 
              << input_image[0][0][0].size() << std::endl;
    
    // swish_block0: 1*3*640*640 -> 1*16*320*320 (stride=2)
    auto output0 = swish_block(input_image);
    // 释放input_image内存
    input_image.clear();
    input_image.shrink_to_fit();
    
    // swish_block1: 1*16*320*320 -> 1*32*160*160 (stride=2)
    auto output1 = swish_block1(output0);
    // 释放output0内存
    output0.clear();
    output0.shrink_to_fit();
    
    // swish_block2: 1*32*160*160 -> 1*32*160*160 (stride=1, 1x1 conv)
    auto output2 = swish_block2(output1);
    // 释放output1内存
    output1.clear();
    output1.shrink_to_fit();
    
    // complex_module0: 1*32*160*160 -> 1*48*160*160 (split + processing + concat)
    auto complex_output_between = complex_module0(output2);
    // 释放output2内存
    output2.clear();
    output2.shrink_to_fit();

    // swish_block2_cv2: 1*48*160*160 -> 1*32*160*160 (1x1 conv)
    auto output2_cv2 = swish_block2_cv2(complex_output_between);
    // 释放complex_output_between内存
    complex_output_between.clear();
    complex_output_between.shrink_to_fit();
    
    // swish_block3: 1*32*160*160 -> 1*64*80*80 (stride=2)
    auto output3 = swish_block3(output2_cv2);
    // 释放output2_cv2内存
    output2_cv2.clear();
    output2_cv2.shrink_to_fit();
    
    // swish_block4_cv1: 1*64*80*80 -> 1*64*80*80 (1x1 conv)
    auto output4_cv1 = swish_block4_cv1(output3);
    // 释放output3内存
    output3.clear();
    output3.shrink_to_fit();
    
    // more_complex_module0: 1*64*80*80 -> 1*128*80*80 (split + processing + concat)
    auto complex_output0 = more_complex_module0(output4_cv1);
    // 释放output4_cv1内存
    output4_cv1.clear();
    output4_cv1.shrink_to_fit();

    // swish_block4_cv2: 1*128*80*80 -> 1*64*80*80 (1x1 conv)
    auto output4_cv2 = swish_block4_cv2(complex_output0);
    
    // 释放complex_output0内存
    complex_output0.clear();
    complex_output0.shrink_to_fit();
    
    // 分成两股
    auto branch1 = output4_cv2;  // 第一股：保留原始数据
    auto branch2 = output4_cv2;  // 第二股：进行后续处理
    // 释放output4_cv2内存
    output4_cv2.clear();
    output4_cv2.shrink_to_fit();
    
    // 第二股处理：swish_block5: 1*64*80*80 -> 1*128*40*40 (stride=2)
    auto output5 = swish_block5(branch2);
    // 释放branch2内存
    branch2.clear();
    branch2.shrink_to_fit();
    
    // swish_block6_cv1: 1*128*40*40 -> 1*128*40*40 (1x1 conv)
    auto output6_cv1 = swish_block6_cv1(output5);
    // 释放output5内存
    output5.clear();
    output5.shrink_to_fit();
    
    // more_complex_module1: 1*128*40*40 -> 1*256*40*40 (split + processing + concat)
    auto complex_output1 = more_complex_module1(output6_cv1);
    // 释放output6_cv1内存
    output6_cv1.clear();
    output6_cv1.shrink_to_fit();
    
    // swish_block6_cv2: 1*256*40*40 -> 1*128*40*40 (1x1 conv)
    auto output6_cv2 = swish_block6_cv2(complex_output1);
    // 释放complex_output1内存
    complex_output1.clear();
    complex_output1.shrink_to_fit();
    
    // 第二次分成两股
    auto branch2_1 = output6_cv2;  // 第一股：保留原始数据
    auto branch2_2 = output6_cv2;  // 第二股：进行后续处理
    // 释放output6_cv2内存
    output6_cv2.clear();
    output6_cv2.shrink_to_fit();
    
    // 第二股处理：swish_block7: 1*128*40*40 -> 1*256*20*20 (stride=2)
    auto output7 = swish_block7(branch2_2);
    // 释放branch2_2内存
    branch2_2.clear();
    branch2_2.shrink_to_fit();
    
    // swish_block8_cv1: 1*256*20*20 -> 1*256*20*20 (1x1 conv)
    auto output8_cv1 = swish_block8_cv1(output7);
    // 释放output7内存
    output7.clear();
    output7.shrink_to_fit();
    
    // complex_module1: 1*256*20*20 -> 1*384*20*20 (split + processing + concat)
    auto complex_output_final = complex_module1(output8_cv1);
    // 释放output8_cv1内存
    output8_cv1.clear();
    output8_cv1.shrink_to_fit();
    
    // swish_block8_cv2: 1*384*20*20 -> 1*256*20*20 (1x1 conv)
    auto output8_cv2 = swish_block8_cv2(complex_output_final);
    // 释放complex_output_final内存
    complex_output_final.clear();
    complex_output_final.shrink_to_fit();
    
    // swish_block9_cv1: 1*256*20*20 -> 1*128*20*20 (1x1 conv)
    auto output9_cv1 = swish_block9_cv1(output8_cv2);
    // 释放output8_cv2内存
    output8_cv2.clear();
    output8_cv2.shrink_to_fit();
    
    // 分成两份
    auto concat_input1 = output9_cv1;  // 第一份：直接连接到concat
    auto maxpool_input = output9_cv1;  // 第二份：经过连续maxpool
    // 释放output9_cv1内存
    output9_cv1.clear();
    output9_cv1.shrink_to_fit();
    
    // 连续三次maxpool (kernel=5x5, stride=1x1, padding=2x2)
    auto maxpool1 = maxpool(maxpool_input, 5, 1, 2);  // 1*128*20*20 (保持尺寸不变)
    // 释放maxpool_input内存
    maxpool_input.clear();
    maxpool_input.shrink_to_fit();
    
    auto maxpool2 = maxpool(maxpool1, 5, 1, 2);       // 1*128*20*20 (保持尺寸不变)
    auto maxpool3 = maxpool(maxpool2, 5, 1, 2);       // 1*128*20*20 (保持尺寸不变)

    // 准备concat的四份输入 (每份都是1*128*20*20)
    std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> concat_inputs;
    concat_inputs.push_back(concat_input1);  // 1*128*20*20
    concat_inputs.push_back(maxpool1);       // 1*128*20*20
    concat_inputs.push_back(maxpool2);       // 1*128*20*20
    concat_inputs.push_back(maxpool3);       // 1*128*20*20
    
    // Concat操作：沿通道轴(axis=1)拼接 -> 1*512*20*20
    auto concat_output = concat(concat_inputs, 1);
    // 释放concat_inputs中的数据
    for(auto& input : concat_inputs) {
        input.clear();
        input.shrink_to_fit();
    }
    concat_inputs.clear();
    concat_inputs.shrink_to_fit();
    
    // swish_block9_cv2: 1*512*20*20 -> 1*256*20*20 (1x1 conv)
    auto output9_cv2 = swish_block9_cv2(concat_output);
    // 释放concat_output内存
    concat_output.clear();
    concat_output.shrink_to_fit();
    
    // 第三次分成两股
    auto branch3_1 = output9_cv2;  // 第一股：保留原始数据
    auto branch3_2 = output9_cv2;  // 第二股：进行后续处理
    // 释放output9_cv2内存
    output9_cv2.clear();
    output9_cv2.shrink_to_fit();
    
    // branch3_1经过resize: 1*256*20*20 -> 1*256*40*40 (scale factor [1,1,2,2])
    auto resized_branch3_1 = resize(
        branch3_1, 40, 40, 
        ResizeMode::NEAREST, 
        CoordinateTransformMode::ASYMMETRIC, 
        NearestMode::FLOOR
    );
    // 释放branch3_1内存
    branch3_1.clear();
    branch3_1.shrink_to_fit();
    
    // branch2_1与resized_branch3_1进行concat
    std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> concat_inputs2;
    concat_inputs2.push_back(resized_branch3_1);  // 1*256*40*40
    concat_inputs2.push_back(branch2_1);          // 1*128*40*40

    
    // Concat操作：沿通道轴(axis=1)拼接 -> 1*384*40*40
    auto concat_output2 = concat(concat_inputs2, 1);
    
    // 释放concat_inputs2中的数据
    branch2_1.clear();
    branch2_1.shrink_to_fit();
    resized_branch3_1.clear();
    resized_branch3_1.shrink_to_fit();
    concat_inputs2.clear();
    concat_inputs2.shrink_to_fit();
    
    // swish_block12_cv1: 1*384*40*40 -> 1*128*40*40 (1x1 conv)
    auto output12_cv1 = swish_block12_cv1(concat_output2);
    // 释放concat_output2内存
    concat_output2.clear();
    concat_output2.shrink_to_fit();
    
    // simple_module0: 1*128*40*40 -> 1*192*40*40 (split + processing + concat)
    auto simple_output0 = simple_module3(output12_cv1);
    // 释放output12_cv1内存
    output12_cv1.clear();
    output12_cv1.shrink_to_fit();
    
    // swish_block12_cv2: 1*192*40*40 -> 1*128*40*40 (1x1 conv)
    auto output12_cv2 = swish_block12_cv2(simple_output0);
    // 释放simple_output0内存
    simple_output0.clear();
    simple_output0.shrink_to_fit();
    
    // 第四次分成两股
    auto branch4_1 = output12_cv2;  // 第一股：进行resize
    auto branch4_2 = output12_cv2;  // 第二股：保留原始数据
    // 释放output12_cv2内存
    output12_cv2.clear();
    output12_cv2.shrink_to_fit();
    
    // branch4_1经过resize: 1*128*40*40 -> 1*128*80*80 (scale factor [1,1,2,2])
    auto resized_branch4_1 = resize(
        branch4_1, 80, 80, 
        ResizeMode::NEAREST, 
        CoordinateTransformMode::ASYMMETRIC, 
        NearestMode::FLOOR
    );
    // 释放branch4_1内存
    branch4_1.clear();
    branch4_1.shrink_to_fit();
    
    // branch1与resized_branch4_1进行concat
    std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> concat_inputs3;
    concat_inputs3.push_back(resized_branch4_1);  // 1*128*80*80
    concat_inputs3.push_back(branch1);            // 1*64*80*80

    
    // Concat操作：沿通道轴(axis=1)拼接 -> 1*192*80*80
    auto concat_output3 = concat(concat_inputs3, 1);
    
    // 释放concat_inputs3中的数据
    branch1.clear();
    branch1.shrink_to_fit();
    resized_branch4_1.clear();
    resized_branch4_1.shrink_to_fit();
    concat_inputs3.clear();
    concat_inputs3.shrink_to_fit();
    
    // swish_block15_cv1: 1*192*80*80 -> 1*64*80*80 (1x1 conv)
    auto output15_cv1 = swish_block15_cv1(concat_output3);
    // 释放concat_output3内存
    concat_output3.clear();
    concat_output3.shrink_to_fit();
    
    // simple_module1: 1*64*80*80 -> 1*96*80*80 (split + processing + concat)
    auto simple_output1 = simple_module0(output15_cv1);
    // 释放output15_cv1内存
    output15_cv1.clear();
    output15_cv1.shrink_to_fit();
    
    // swish_block15_cv2: 1*96*80*80 -> 1*64*80*80 (1x1 conv)
    auto output15_cv2 = swish_block15_cv2(simple_output1);
    // 释放simple_output1内存
    simple_output1.clear();
    simple_output1.shrink_to_fit();
    //
    // 分成三股
    auto branch5_1 = output15_cv2;  // 第一股：经过sb22 cv2系列
    auto branch5_2 = output15_cv2;  // 第二股：经过sb22 cv3系列
    auto branch5_3 = output15_cv2;  // 第三股：保留原始数据
    // 释放output15_cv2内存
    output15_cv2.clear();
    output15_cv2.shrink_to_fit();
    
    // 第一股处理：swish_block22_cv2_0_0 和 swish_block22_cv2_0_1
    auto output22_cv2_0_0 = swish_block22_cv2_0_0(branch5_1);  // 1*64*80*80 -> 1*64*80*80
    // 释放branch5_1内存
    branch5_1.clear();
    branch5_1.shrink_to_fit();
    
    auto output22_cv2_0_1 = swish_block22_cv2_0_1(output22_cv2_0_0);  // 1*64*80*80 -> 1*64*80*80
    // 释放output22_cv2_0_0内存
    output22_cv2_0_0.clear();
    output22_cv2_0_0.shrink_to_fit();
    
    // conv22_cv2_0_2: 1*64*80*80 -> 1*64*80*80 (1x1 conv, stride=1, padding=0)
    // 权重维度转换: [64*64*1*1] -> [64][64][1][1]
    std::vector<std::vector<std::vector<std::vector<float>>>> conv22_cv2_0_2_weight(
        64, std::vector<std::vector<std::vector<float>>>(
            64, std::vector<std::vector<float>>(
                1, std::vector<float>(1)
            )
        )
    );
    int idx_cv2_0_2 = 0;
    for (int oc = 0; oc < 64; oc++) {
        for (int ic = 0; ic < 64; ic++) {
            for (int kh = 0; kh < 1; kh++) {
                for (int kw = 0; kw < 1; kw++) {
                    conv22_cv2_0_2_weight[oc][ic][kh][kw] = model_22_cv2_0_2_weight[idx_cv2_0_2++];
                }
            }
        }
    }
    
    // 偏置维度转换: [64] -> [64]
    std::vector<float> conv22_cv2_0_2_bias(64);
    for (int i = 0; i < 64; i++) {
        conv22_cv2_0_2_bias[i] = model_22_cv2_0_2_bias[i];
    }
    
    auto output22_cv2_0_2 = conv(output22_cv2_0_1, 
                                        conv22_cv2_0_2_weight, 
                                        conv22_cv2_0_2_bias,
                                        {1, 1}, {0, 0});
    // 释放output22_cv2_0_1内存
    output22_cv2_0_1.clear();
    output22_cv2_0_1.shrink_to_fit();
    
    // 第二股处理：swish_block22_cv3_0_0 和 swish_block22_cv3_0_1
    auto output22_cv3_0_0 = swish_block22_cv3_0_0(branch5_2);  // 1*64*80*80 -> 1*64*80*80
    // 释放branch5_2内存
    branch5_2.clear();
    branch5_2.shrink_to_fit();
    
    auto output22_cv3_0_1 = swish_block22_cv3_0_1(output22_cv3_0_0);  // 1*64*80*80 -> 1*64*80*80
    // 释放output22_cv3_0_0内存
    output22_cv3_0_0.clear();
    output22_cv3_0_0.shrink_to_fit();
    
    // conv22_cv3_0_2: 1*64*80*80 -> 1*3*80*80 (1x1 conv, stride=1, padding=0)
    // 权重维度转换: [3*64*1*1] -> [3][64][1][1]
    std::vector<std::vector<std::vector<std::vector<float>>>> conv22_cv3_0_2_weight(
        3, std::vector<std::vector<std::vector<float>>>(
            64, std::vector<std::vector<float>>(
                1, std::vector<float>(1)
            )
        )
    );
    int idx_cv3_0_2 = 0;
    for (int oc = 0; oc < 3; oc++) {
        for (int ic = 0; ic < 64; ic++) {
            for (int kh = 0; kh < 1; kh++) {
                for (int kw = 0; kw < 1; kw++) {
                    conv22_cv3_0_2_weight[oc][ic][kh][kw] = model_22_cv3_0_2_weight[idx_cv3_0_2++];
                }
            }
        }
    }
    
    // 偏置维度转换: [3] -> [3]
    std::vector<float> conv22_cv3_0_2_bias(3);
    for (int i = 0; i < 3; i++) {
        conv22_cv3_0_2_bias[i] = model_22_cv3_0_2_bias[i];
    }
    
    auto output22_cv3_0_2 = conv(output22_cv3_0_1,
                                        conv22_cv3_0_2_weight,
                                        conv22_cv3_0_2_bias,
                                        {1, 1}, {0, 0});
    // 释放output22_cv3_0_1内存
    output22_cv3_0_1.clear();
    output22_cv3_0_1.shrink_to_fit();
    
    // 第三股处理：branch5_3经过swish_block16
    auto output16 = swish_block16(branch5_3);  // 1*64*80*80 -> 1*64*40*40 (stride=2, padding=1)
    // 释放branch5_3内存
    branch5_3.clear();
    branch5_3.shrink_to_fit();
    
    // branch4_2与处理后的branch5_3进行concat
    std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> concat_inputs_branch4;
    concat_inputs_branch4.push_back(output16);                  // 1*64*40*40
    concat_inputs_branch4.push_back(branch4_2);                 // 1*128*40*40

    
    // Concat操作 -> 1*192*40*40
    auto concat_output_branch4 = concat(concat_inputs_branch4, 1);
    // 释放concat_inputs_branch4中的数据
    branch4_2.clear();
    branch4_2.shrink_to_fit();
    output16.clear();
    output16.shrink_to_fit();
    concat_inputs_branch4.clear();
    concat_inputs_branch4.shrink_to_fit();
    
    // swish_block18_cv1: 1*192*40*40 -> 1*128*40*40 (1x1 conv)
    auto output18_cv1 = swish_block18_cv1(concat_output_branch4);
    // 释放concat_output_branch4内存
    concat_output_branch4.clear();
    concat_output_branch4.shrink_to_fit();
    
    // simple_module2: 1*128*40*40 -> 1*192*40*40 (split + processing + concat)
    auto simple_output2 = simple_module1(output18_cv1);
    // 释放output18_cv1内存
    output18_cv1.clear();
    output18_cv1.shrink_to_fit();
    
    // swish_block18_cv2: 1*192*40*40 -> 1*128*40*40 (1x1 conv)
    auto output18_cv2 = swish_block18_cv2(simple_output2);
    // 释放simple_output2内存
    simple_output2.clear();
    simple_output2.shrink_to_fit();
    
    // 分成三股
    auto branch6_1 = output18_cv2;  // 第一股：经过sb22 cv2_1系列
    auto branch6_2 = output18_cv2;  // 第二股：经过sb22 cv3_1系列
    auto branch6_3 = output18_cv2;  // 第三股：保留原始数据
    // 释放output18_cv2内存
    output18_cv2.clear();
    output18_cv2.shrink_to_fit();
    
    // 第一股处理：swish_block22_cv2_1_0 和 swish_block22_cv2_1_1
    auto output22_cv2_1_0 = swish_block22_cv2_1_0(branch6_1);  // 1*128*40*40 -> 1*64*40*40
    // 释放branch6_1内存
    branch6_1.clear();
    branch6_1.shrink_to_fit();
    
    auto output22_cv2_1_1 = swish_block22_cv2_1_1(output22_cv2_1_0);  // 1*64*40*40 -> 1*64*40*40
    // 释放output22_cv2_1_0内存
    output22_cv2_1_0.clear();
    output22_cv2_1_0.shrink_to_fit();
    
    // conv22_cv2_1_2: 1*64*40*40 -> 1*64*40*40 (1x1 conv, stride=1, padding=0)
    // 权重维度转换: [64*64*1*1] -> [64][64][1][1]
    std::vector<std::vector<std::vector<std::vector<float>>>> conv22_cv2_1_2_weight(
        64, std::vector<std::vector<std::vector<float>>>(
            64, std::vector<std::vector<float>>(
                1, std::vector<float>(1)
            )
        )
    );
    int idx_cv2_1_2 = 0;
    for (int oc = 0; oc < 64; oc++) {
        for (int ic = 0; ic < 64; ic++) {
            for (int kh = 0; kh < 1; kh++) {
                for (int kw = 0; kw < 1; kw++) {
                    conv22_cv2_1_2_weight[oc][ic][kh][kw] = model_22_cv2_1_2_weight[idx_cv2_1_2++];
                }
            }
        }
    }
    
    // 偏置维度转换: [64] -> [64]
    std::vector<float> conv22_cv2_1_2_bias(64);
    for (int i = 0; i < 64; i++) {
        conv22_cv2_1_2_bias[i] = model_22_cv2_1_2_bias[i];
    }
    
    auto output22_cv2_1_2 = conv(output22_cv2_1_1, 
                                        conv22_cv2_1_2_weight, 
                                        conv22_cv2_1_2_bias,
                                        {1, 1}, {0, 0});
    // 释放output22_cv2_1_1内存
    output22_cv2_1_1.clear();
    output22_cv2_1_1.shrink_to_fit();
    
    // 第二股处理：swish_block22_cv3_1_0 和 swish_block22_cv3_1_1
    auto output22_cv3_1_0 = swish_block22_cv3_1_0(branch6_2);  // 1*128*40*40 -> 1*64*40*40
    // 释放branch6_2内存
    branch6_2.clear();
    branch6_2.shrink_to_fit();
    
    auto output22_cv3_1_1 = swish_block22_cv3_1_1(output22_cv3_1_0);  // 1*64*40*40 -> 1*64*40*40
    // 释放output22_cv3_1_0内存
    output22_cv3_1_0.clear();
    output22_cv3_1_0.shrink_to_fit();
    
    // conv22_cv3_1_2: 1*64*40*40 -> 1*3*40*40 (1x1 conv, stride=1, padding=0)
    // 权重维度转换: [3*64*1*1] -> [3][64][1][1]
    std::vector<std::vector<std::vector<std::vector<float>>>> conv22_cv3_1_2_weight(
        3, std::vector<std::vector<std::vector<float>>>(
            64, std::vector<std::vector<float>>(
                1, std::vector<float>(1)
            )
        )
    );
    int idx_cv3_1_2 = 0;
    for (int oc = 0; oc < 3; oc++) {
        for (int ic = 0; ic < 64; ic++) {
            for (int kh = 0; kh < 1; kh++) {
                for (int kw = 0; kw < 1; kw++) {
                    conv22_cv3_1_2_weight[oc][ic][kh][kw] = model_22_cv3_1_2_weight[idx_cv3_1_2++];
                }
            }
        }
    }
    
    // 偏置维度转换: [3] -> [3]
    std::vector<float> conv22_cv3_1_2_bias(3);
    for (int i = 0; i < 3; i++) {
        conv22_cv3_1_2_bias[i] = model_22_cv3_1_2_bias[i];
    }
    
    auto output22_cv3_1_2 = conv(output22_cv3_1_1,
                                        conv22_cv3_1_2_weight,
                                        conv22_cv3_1_2_bias,
                                        {1, 1}, {0, 0});
    // 释放output22_cv3_1_1内存
    output22_cv3_1_1.clear();
    output22_cv3_1_1.shrink_to_fit();
    
    // 将branch6的两股数据concat：沿通道轴(axis=1)拼接
    std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> branch6_concat_inputs;
    branch6_concat_inputs.push_back(output22_cv2_1_2);  // 1*64*40*40
    branch6_concat_inputs.push_back(output22_cv3_1_2);  // 1*3*40*40
    
    // Concat操作 -> 1*67*40*40
    auto branch6_concat_output = concat(branch6_concat_inputs, 1);
    // 释放branch6_concat_inputs中的数据
    output22_cv2_1_2.clear();
    output22_cv2_1_2.shrink_to_fit();
    output22_cv3_1_2.clear();
    output22_cv3_1_2.shrink_to_fit();
    branch6_concat_inputs.clear();
    branch6_concat_inputs.shrink_to_fit();
    
    // Reshape: 1*67*40*40 -> 1*67*1600 (40*40 = 1600)
    std::vector<int> branch6_target_shape = {1, 67, 1600};
    auto branch6_reshape_output = reshape_4d_to_3d(branch6_concat_output, branch6_target_shape);
    // 释放branch6_concat_output内存
    branch6_concat_output.clear();
    branch6_concat_output.shrink_to_fit();

    // 将两股数据concat：沿通道轴(axis=1)拼接
    std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> branch5_concat_inputs;
    branch5_concat_inputs.push_back(output22_cv2_0_2);  // 1*64*80*80
    branch5_concat_inputs.push_back(output22_cv3_0_2);  // 1*3*80*80
    
    // Concat操作 -> 1*67*80*80
    auto branch5_concat_output = concat(branch5_concat_inputs, 1);
    // 释放branch5_concat_inputs中的数据
    output22_cv2_0_2.clear();
    output22_cv2_0_2.shrink_to_fit();
    output22_cv3_0_2.clear();
    output22_cv3_0_2.shrink_to_fit();
    branch5_concat_inputs.clear();
    branch5_concat_inputs.shrink_to_fit();
    
    // Reshape: 1*67*80*80 -> 1*67*6400 (80*80 = 6400)
    std::vector<int> branch5_target_shape = {1, 67, 6400};
    auto branch5_reshape_output = reshape_4d_to_3d(branch5_concat_output, branch5_target_shape);
    // 释放branch5_concat_output内存
    branch5_concat_output.clear();
    branch5_concat_output.shrink_to_fit();

    // 第三股处理：branch6_3经过swish_block19
    auto output19 = swish_block19(branch6_3);  // 1*128*40*40 -> 1*128*20*20 (stride=2, padding=1)
    // 释放branch6_3内存
    branch6_3.clear();
    branch6_3.shrink_to_fit();
    
    // branch3_2与处理后的branch6_3进行concat
    
    std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> concat_inputs_branch3;
    concat_inputs_branch3.push_back(output19);                  // 1*128*20*20
    concat_inputs_branch3.push_back(branch3_2);                 // 1*256*20*20

    
    // Concat操作 -> 1*384*20*20
    auto concat_output_branch3 = concat(concat_inputs_branch3, 1);
    
    // 释放concat_inputs_branch3中的数据
    branch3_2.clear();
    branch3_2.shrink_to_fit();
    output19.clear();
    output19.shrink_to_fit();
    concat_inputs_branch3.clear();
    concat_inputs_branch3.shrink_to_fit();

    // swish_block21_cv1: 1*384*20*20 -> 1*256*20*20 (1x1 conv)
    auto output21_cv1 = swish_block21_cv1(concat_output_branch3);
    // 释放concat_output_branch3内存
    concat_output_branch3.clear();
    concat_output_branch3.shrink_to_fit();
    
    // simple_module3: 1*256*20*20 -> 1*320*20*20 (split + processing + concat)
    auto simple_output3 = simple_module2(output21_cv1);
    // 释放output21_cv1内存
    output21_cv1.clear();
    output21_cv1.shrink_to_fit();
    
    // swish_block21_cv2: 1*320*20*20 -> 1*256*20*20 (1x1 conv)
    auto output21_cv2 = swish_block21_cv2(simple_output3);
    // 释放simple_output3内存
    simple_output3.clear();
    simple_output3.shrink_to_fit();

    // 分成两股
    auto branch7_1 = output21_cv2;  // 第一股：经过sb22 cv2_2系列
    auto branch7_2 = output21_cv2;  // 第二股：经过sb22 cv3_2系列
    // 释放output21_cv2内存
    output21_cv2.clear();
    output21_cv2.shrink_to_fit();
    
    // 第一股处理：swish_block22_cv2_2_0 和 swish_block22_cv2_2_1
    auto output22_cv2_2_0 = swish_block22_cv2_2_0(branch7_1);  // 1*256*20*20 -> 1*64*20*20
    // 释放branch7_1内存
    branch7_1.clear();
    branch7_1.shrink_to_fit();
    
    auto output22_cv2_2_1 = swish_block22_cv2_2_1(output22_cv2_2_0);  // 1*64*20*20 -> 1*64*20*20
    // 释放output22_cv2_2_0内存
    output22_cv2_2_0.clear();
    output22_cv2_2_0.shrink_to_fit();
    
    // conv22_cv2_2_2: 1*64*20*20 -> 1*64*20*20 (1x1 conv, stride=1, padding=0)
    // 权重维度转换: [64*64*1*1] -> [64][64][1][1]
    std::vector<std::vector<std::vector<std::vector<float>>>> conv22_cv2_2_2_weight(
        64, std::vector<std::vector<std::vector<float>>>(
            64, std::vector<std::vector<float>>(
                1, std::vector<float>(1)
            )
        )
    );
    int idx_cv2_2_2 = 0;
    for (int oc = 0; oc < 64; oc++) {
        for (int ic = 0; ic < 64; ic++) {
            for (int kh = 0; kh < 1; kh++) {
                for (int kw = 0; kw < 1; kw++) {
                    conv22_cv2_2_2_weight[oc][ic][kh][kw] = model_22_cv2_2_2_weight[idx_cv2_2_2++];
                }
            }
        }
    }
    
    // 偏置维度转换: [64] -> [64]
    std::vector<float> conv22_cv2_2_2_bias(64);
    for (int i = 0; i < 64; i++) {
        conv22_cv2_2_2_bias[i] = model_22_cv2_2_2_bias[i];
    }
    
    auto output22_cv2_2_2 = conv(output22_cv2_2_1, 
                                        conv22_cv2_2_2_weight, 
                                        conv22_cv2_2_2_bias,
                                        {1, 1}, {0, 0});
    // 释放output22_cv2_2_1内存
    output22_cv2_2_1.clear();
    output22_cv2_2_1.shrink_to_fit();
    
    // 第二股处理：swish_block22_cv3_2_0 和 swish_block22_cv3_2_1
    auto output22_cv3_2_0 = swish_block22_cv3_2_0(branch7_2);  // 1*256*20*20 -> 1*64*20*20
    // 释放branch7_2内存
    branch7_2.clear();
    branch7_2.shrink_to_fit();
    
    auto output22_cv3_2_1 = swish_block22_cv3_2_1(output22_cv3_2_0);  // 1*64*20*20 -> 1*64*20*20
    // 释放output22_cv3_2_0内存
    output22_cv3_2_0.clear();
    output22_cv3_2_0.shrink_to_fit();
    
    // conv22_cv3_2_2: 1*64*20*20 -> 1*3*20*20 (1x1 conv, stride=1, padding=0)
    // 权重维度转换: [3*64*1*1] -> [3][64][1][1]
    std::vector<std::vector<std::vector<std::vector<float>>>> conv22_cv3_2_2_weight(
        3, std::vector<std::vector<std::vector<float>>>(
            64, std::vector<std::vector<float>>(
                1, std::vector<float>(1)
            )
        )
    );
    int idx_cv3_2_2 = 0;
    for (int oc = 0; oc < 3; oc++) {
        for (int ic = 0; ic < 64; ic++) {
            for (int kh = 0; kh < 1; kh++) {
                for (int kw = 0; kw < 1; kw++) {
                    conv22_cv3_2_2_weight[oc][ic][kh][kw] = model_22_cv3_2_2_weight[idx_cv3_2_2++];
                }
            }
        }
    }
    
    // 偏置维度转换: [3] -> [3]
    std::vector<float> conv22_cv3_2_2_bias(3);
    for (int i = 0; i < 3; i++) {
        conv22_cv3_2_2_bias[i] = model_22_cv3_2_2_bias[i];
    }
    
    auto output22_cv3_2_2 = conv(output22_cv3_2_1,
                                        conv22_cv3_2_2_weight,
                                        conv22_cv3_2_2_bias,
                                        {1, 1}, {0, 0});
    // 释放output22_cv3_2_1内存
    output22_cv3_2_1.clear();
    output22_cv3_2_1.shrink_to_fit();
    
    // 将branch7的两股数据concat：沿通道轴(axis=1)拼接
    std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> branch7_concat_inputs;
    branch7_concat_inputs.push_back(output22_cv2_2_2);  // 1*64*20*20
    branch7_concat_inputs.push_back(output22_cv3_2_2);  // 1*3*20*20
    
    // Concat操作 -> 1*67*20*20
    auto branch7_concat_output = concat(branch7_concat_inputs, 1);
    // 释放branch7_concat_inputs中的数据
    output22_cv2_2_2.clear();
    output22_cv2_2_2.shrink_to_fit();
    output22_cv3_2_2.clear();
    output22_cv3_2_2.shrink_to_fit();
    branch7_concat_inputs.clear();
    branch7_concat_inputs.shrink_to_fit();
    
    // Reshape: 1*67*20*20 -> 1*67*400 (20*20 = 400)
    std::vector<int> branch7_target_shape = {1, 67, 400};
    auto branch7_reshape_output = reshape_4d_to_3d(branch7_concat_output, branch7_target_shape);
    // 释放branch7_concat_output内存
    branch7_concat_output.clear();
    branch7_concat_output.shrink_to_fit();

    std::vector<std::vector<std::vector<std::vector<float>>>> all_outputs_concat_inputs;
    all_outputs_concat_inputs.push_back(branch5_reshape_output);  // 1*67*6400
    all_outputs_concat_inputs.push_back(branch6_reshape_output);  // 1*67*1600
    all_outputs_concat_inputs.push_back(branch7_reshape_output);  // 1*67*400
    
    // 最终concat操作 -> 1*67*8400 (6400+1600+400 = 8400)
    auto all_outputs_concat = concat_3d(all_outputs_concat_inputs, 2);
    
    // 释放all_outputs_concat_inputs中的数据
    branch5_reshape_output.clear();
    branch5_reshape_output.shrink_to_fit();
    branch6_reshape_output.clear();
    branch6_reshape_output.shrink_to_fit();
    branch7_reshape_output.clear();
    branch7_reshape_output.shrink_to_fit();
    all_outputs_concat_inputs.clear();
    all_outputs_concat_inputs.shrink_to_fit();
    
    // Split操作: 1*67*8400 -> [1*64*8400, 1*3*8400]
    std::vector<int> split_sizes = {64, 3};
    auto split_outputs = split_3d(all_outputs_concat, 1, split_sizes);
    // 释放all_outputs_concat内存
    all_outputs_concat.clear();
    all_outputs_concat.shrink_to_fit();
    
    // 获取split后的两个输出
    auto bbox_output = split_outputs[0];  // 1*64*8400
    auto cls_output = split_outputs[1];   // 1*3*8400
    // 释放split_outputs内存
    split_outputs.clear();
    split_outputs.shrink_to_fit();

    // 使用reshape函数: 1*64*1*8400 -> 按[1,4,16,8400]重塑为4D
    std::vector<int> bbox_target_shape = {1, 4, 16, 8400};
    auto bbox_reshaped = reshape_3d_to_4d(bbox_output, bbox_target_shape);
    // 释放bbox_output内存
    bbox_output.clear();
    bbox_output.shrink_to_fit();
    
    // Transpose操作: 1*4*16*8400 -> 1*16*4*8400
    std::vector<int> transpose_axes = {0, 2, 1, 3};
    auto bbox_transposed = transpose(bbox_reshaped, transpose_axes);
    // 释放bbox_reshaped内存
    bbox_reshaped.clear();
    bbox_reshaped.shrink_to_fit();
    
    // Softmax操作: 1*16*4*8400 -> 1*16*4*8400 (axis=1应用softmax，维度保持不变)
    auto bbox_softmax = softmax(bbox_transposed, 1);
    // 释放bbox_transposed内存
    bbox_transposed.clear();
    bbox_transposed.shrink_to_fit();
    
    // DFL卷积操作: 1*16*4*8400 -> 1*1*4*8400 
    // 权重维度转换: [1*16*1*1] -> [1][16][1][1]
    std::vector<std::vector<std::vector<std::vector<float>>>> dfl_conv_weight(
        1, std::vector<std::vector<std::vector<float>>>(
            16, std::vector<std::vector<float>>(
                1, std::vector<float>(1)
            )
        )
    );
    int idx_dfl = 0;
    for (int oc = 0; oc < 1; oc++) {
        for (int ic = 0; ic < 16; ic++) {
            for (int kh = 0; kh < 1; kh++) {
                for (int kw = 0; kw < 1; kw++) {
                    dfl_conv_weight[oc][ic][kh][kw] = model_22_dfl_conv_weight[idx_dfl++];
                }
            }
        }
    }
    
    // DFL卷积通常没有bias，创建零bias
    std::vector<float> dfl_conv_bias(1, 0.0f);
    
    auto bbox_dfl = conv(bbox_softmax,
                         dfl_conv_weight,
                         dfl_conv_bias, 
                         {1, 1}, {0, 0});
    // 释放bbox_softmax内存
    bbox_softmax.clear();
    bbox_softmax.shrink_to_fit();
    
    // Reshape操作: 1*1*4*8400 -> 1*4*8400 (去掉第二个维度)
    std::vector<int> bbox_final_shape = {1, 4, 8400};
    auto bbox_final = reshape_4d_to_3d(bbox_dfl, bbox_final_shape);
    // 释放bbox_dfl内存
    bbox_dfl.clear();
    bbox_dfl.shrink_to_fit();
    
    // bbox_final分成两股进行slice操作
    auto bbox_branch1 = bbox_final;  // 第一股
    auto bbox_branch2 = bbox_final;  // 第二股
    // 释放bbox_final内存
    bbox_final.clear();
    bbox_final.shrink_to_fit();
    
    // 第一股slice操作: 1*4*8400 -> 1*2*8400 (start=0, end=2, axis=1)
    auto bbox_slice1 = slice(bbox_branch1, 0, 2, 1);  // 取前2个通道
    // 释放bbox_branch1内存
    bbox_branch1.clear();
    bbox_branch1.shrink_to_fit();
    
    // 第二股slice操作: 1*4*8400 -> 1*2*8400 (start=2, end=4, axis=1)
    auto bbox_slice2 = slice(bbox_branch2, 2, 4, 1);  // 取后2个通道
    // 释放bbox_branch2内存
    bbox_branch2.clear();
    bbox_branch2.shrink_to_fit();
    
    // bbox_slice1进行sub操作，减去model_22_Constant_9_output_0 (A)
    // 将model_22_Constant_9_output_0 [16800] 转换为 1*2*8400
    std::vector<std::vector<std::vector<float>>> constant_tensor(
        1, std::vector<std::vector<float>>(
            2, std::vector<float>(8400)
        )
    );
    
    // 填充数据：前8400个元素给第一个通道，后8400个给第二个通道
    for (int i = 0; i < 8400; i++) {
        constant_tensor[0][0][i] = model_22_Constant_9_output_0[i];        // 第一个通道
        constant_tensor[0][1][i] = model_22_Constant_9_output_0[i + 8400]; // 第二个通道
    }
    
    auto bbox_sub = sub_3d(constant_tensor, bbox_slice1);
    // 释放bbox_slice1内存
    bbox_slice1.clear();
    bbox_slice1.shrink_to_fit();
    
    // bbox_slice2进行add_3d操作，加上constant_tensor
    auto bbox_add = add_3d(constant_tensor, bbox_slice2);
    // 释放bbox_slice2内存
    bbox_slice2.clear();
    bbox_slice2.shrink_to_fit();
    
    // bbox_sub和bbox_add分成两股进行进一步处理
    auto bbox_sub_branch1 = bbox_sub;  // 第一股
    auto bbox_sub_branch2 = bbox_sub;  // 第二股
    auto bbox_add_branch1 = bbox_add;  // 第一股
    auto bbox_add_branch2 = bbox_add;  // 第二股
    
    // 第一股：bbox_sub - bbox_add
    auto bbox_final_branch1 = sub_3d(bbox_add_branch1,bbox_sub_branch1);
    // 释放bbox_sub_branch1和bbox_add_branch1内存
    bbox_sub_branch1.clear();
    bbox_sub_branch1.shrink_to_fit();
    bbox_add_branch1.clear();
    bbox_add_branch1.shrink_to_fit();
    
    // 第二股：bbox_sub + bbox_add  
    auto bbox_final_branch2 = add_3d(bbox_sub_branch2, bbox_add_branch2);
    // 释放bbox_sub_branch2和bbox_add_branch2内存
    bbox_sub_branch2.clear();
    bbox_sub_branch2.shrink_to_fit();
    bbox_add_branch2.clear();
    bbox_add_branch2.shrink_to_fit();
    
    // bbox_final_branch2的每个元素除以2

    auto bbox_final_branch2_div = div_3d(bbox_final_branch2,  2.0f);
    // 释放bbox_final_branch2内存
    bbox_final_branch2.clear();
    bbox_final_branch2.shrink_to_fit();
    
    // 将bbox_final_branch1和bbox_final_branch2_div进行concat_3d操作
    std::vector<std::vector<std::vector<std::vector<float>>>> bbox_concat_inputs;
    bbox_concat_inputs.push_back(bbox_final_branch2_div);    // 1*2*8400
    bbox_concat_inputs.push_back(bbox_final_branch1); // 1*2*8400
    
    // 沿第1维(通道维)进行concat -> 1*4*8400
    auto bbox_final_concat = concat_3d(bbox_concat_inputs, 1);
    // 释放bbox_concat_inputs中的数据
    bbox_final_branch2_div.clear();
    bbox_final_branch2_div.shrink_to_fit();
    bbox_final_branch1.clear();
    bbox_final_branch1.shrink_to_fit();
    bbox_concat_inputs.clear();
    bbox_concat_inputs.shrink_to_fit();
    /*
    // 将model_22_Constant_12_output_0 [8400] 转换为 1*1*8400 以便广播
    std::vector<std::vector<std::vector<float>>> constant_12_tensor(
        1, std::vector<std::vector<float>>(
            1, std::vector<float>(8400)
        )
    );
    
    // 填充数据
    for (int i = 0; i < 8400; i++) {
        constant_12_tensor[0][0][i] = model_22_Constant_12_output_0[i];
    }
    
    // bbox_final_concat与model_22_Constant_12_output_0进行mul_3d操作
    auto bbox_mul_result = mul_3d(bbox_final_concat, constant_12_tensor);
    // 释放bbox_final_concat内存
    bbox_final_concat.clear();
    bbox_final_concat.shrink_to_fit();
    
    // cls_output经过sigmoid_3d激活
    auto cls_sigmoid_result = sigmoid_3d(cls_output);
    // 释放cls_output内存
    cls_output.clear();
    cls_output.shrink_to_fit();
    
    // 将cls_sigmoid_result和bbox_mul_result进行concat_3d操作
    std::vector<std::vector<std::vector<std::vector<float>>>> final_concat_inputs;
    final_concat_inputs.push_back(bbox_mul_result);     // 1*4*8400
    final_concat_inputs.push_back(cls_sigmoid_result);  // 1*3*8400
    
    // 沿第1维(通道维)进行concat -> 1*7*8400
    auto final_output = concat_3d(final_concat_inputs, 1);
    // 释放final_concat_inputs中的数据
    bbox_mul_result.clear();
    bbox_mul_result.shrink_to_fit();
    cls_sigmoid_result.clear();
    cls_sigmoid_result.shrink_to_fit();
    final_concat_inputs.clear();
    final_concat_inputs.shrink_to_fit();
    */
    return bbox_final_concat;

    //return final_output;


}



