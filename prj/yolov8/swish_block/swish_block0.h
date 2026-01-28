#ifndef SWISH_BLOCK_H
#define SWISH_BLOCK_H

#include <vector>

// Swish Block: input -> conv -> sigmoid -> multiply
// input: [1, 3, 640, 640]
// output: [1, 16, 320, 320] (assuming stride=2, padding=1, kernel=3x3)
std::vector<std::vector<std::vector<std::vector<float>>>> swish_block(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input
);

#endif
