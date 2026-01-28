#include <vector>
#include <algorithm>

// Element-wise multiplication
// input1: [batch, channels, height, width]
// input2: [batch, channels, height, width] or broadcastable shape
std::vector<std::vector<std::vector<std::vector<float>>>> mul(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input1,
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input2
) {
    int batch = input1.size();
    int channels = input1[0].size();
    int height = input1[0][0].size();
    int width = input1[0][0][0].size();
    
    // Initialize output tensor with same dimensions as input1
    std::vector<std::vector<std::vector<std::vector<float>>>> output(
        batch, std::vector<std::vector<std::vector<float>>>(
            channels, std::vector<std::vector<float>>(
                height, std::vector<float>(width, 0.0f)
            )
        )
    );
    
    // Get dimensions of input2 for broadcasting
    int batch2 = input2.size();
    int channels2 = input2[0].size();
    int height2 = input2[0][0].size();
    int width2 = input2[0][0][0].size();
    
    // Element-wise multiplication with broadcasting
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int b2 = std::min(b, batch2 - 1);
                    int c2 = std::min(c, channels2 - 1);
                    int h2 = std::min(h, height2 - 1);
                    int w2 = std::min(w, width2 - 1);
                    
                    output[b][c][h][w] = input1[b][c][h][w] * input2[b2][c2][h2][w2];
                }
            }
        }
    }
    
    return output;
}
