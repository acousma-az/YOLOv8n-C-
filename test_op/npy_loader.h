#ifndef NPY_LOADER_H
#define NPY_LOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>

class NpyLoader {
public:
    // 加载4D float32张量
    static std::vector<std::vector<std::vector<std::vector<float>>>> 
    load_4d_float32(const std::string& filename);
    
    // 加载3D float32张量  
    static std::vector<std::vector<std::vector<float>>> 
    load_3d_float32(const std::string& filename);
    
    // 跳过numpy头部 (public访问)
    static void skip_npy_header(std::ifstream& file);
    
private:
    // 解析numpy头部信息
    static std::vector<int> parse_npy_header(std::ifstream& file);
};

// 便利的包装函数
std::vector<std::vector<std::vector<float>>> load_3d_tensor(const std::string& filename);
std::vector<std::vector<std::vector<std::vector<float>>>> load_4d_tensor(const std::string& filename);
std::vector<float> load_params(const std::string& filename);

#endif
