#include <iostream>
#include <iomanip>
#include <cmath>

int main() {
    std::cout << "=== 浮点数精度演示 ===" << std::endl;
    std::cout << std::fixed << std::setprecision(10);
    
    float a = 6.0f;
    float b = 6.1f;
    
    std::cout << "a = " << a << std::endl;
    std::cout << "b = " << b << std::endl;
    
    float diff = std::abs(a - b);
    std::cout << "差值 = " << diff << std::endl;
    
    std::cout << "\n理论上应该是 0.1，但实际是 " << diff << std::endl;
    
    std::cout << "\n这是因为：" << std::endl;
    std::cout << "1. 浮点数在计算机中用二进制表示" << std::endl;
    std::cout << "2. 0.1 无法用二进制精确表示" << std::endl;
    std::cout << "3. 类似于十进制中 1/3 = 0.333..." << std::endl;
    
    // 更详细的例子
    std::cout << "\n更多例子：" << std::endl;
    float x = 0.1f;
    std::cout << "0.1f 实际存储为: " << x << std::endl;
    
    float y = 0.2f;
    std::cout << "0.2f 实际存储为: " << y << std::endl;
    
    float z = x + y;
    std::cout << "0.1f + 0.2f = " << z << std::endl;
    std::cout << "应该等于 0.3，但实际是: " << z << std::endl;
    
    if (z == 0.3f) {
        std::cout << "z == 0.3f: true" << std::endl;
    } else {
        std::cout << "z == 0.3f: false (这就是为什么需要容忍度比较！)" << std::endl;
    }
    
    return 0;
}
