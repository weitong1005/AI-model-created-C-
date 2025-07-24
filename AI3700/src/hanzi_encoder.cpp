#include "hanzi_encoder.h"
#include <fstream>
#include <iostream>
#include <cctype>
#include <algorithm>

void HanziEncoder::clear() {
    hanziToId.clear();
    idToHanzi.clear();
}

bool HanziEncoder::loadFromFile(const std::string& filename) {
    clear();
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return false;
    }

    std::string line;
    int totalAdded = 0;
    int lineNumber = 0;

    while (std::getline(infile, line)) {
        lineNumber++;
        if (line.empty()) continue;

        // 提取汉字部分（跳过拼音）
        size_t pos = line.find(':');
        if (pos == std::string::npos) continue;

        std::string hanziBlock = line.substr(pos + 1);

        // 移除所有空格和星号
        hanziBlock.erase(std::remove_if(hanziBlock.begin(), hanziBlock.end(),
                                        [](char c) { return std::isspace(static_cast<unsigned char>(c)) || c == '*'; }),
                         hanziBlock.end());

        // 处理连续UTF-8字符
        for (size_t i = 0; i < hanziBlock.size();) {
            int len = 0;
            unsigned char c = static_cast<unsigned char>(hanziBlock[i]);

            if (c < 0x80) {
                len = 1;  // ASCII字符
            } else if ((c & 0xE0) == 0xC0) {
                len = 2;  // 2字节UTF-8
            } else if ((c & 0xF0) == 0xE0) {
                len = 3;  // 3字节UTF-8（大多数汉字）
            } else if ((c & 0xF8) == 0xF0) {
                len = 4;  // 4字节UTF-8
            } else {
                i++;  // 跳过无效字节
                continue;
            }

            if (i + len > hanziBlock.size()) break;

            std::string hanzi = hanziBlock.substr(i, len);
            i += len;

            // 添加到编码器
            if (hanziToId.find(hanzi) == hanziToId.end()) {
                hanziToId[hanzi] = idToHanzi.size();
                idToHanzi.push_back(hanzi);
                totalAdded++;
            }
        }
    }

    std::cout << "成功加载 " << totalAdded << " 个汉字\n";
    return true;
}

int HanziEncoder::encode(const std::string& hanzi) const {
    auto it = hanziToId.find(hanzi);
    return (it != hanziToId.end()) ? it->second : -1;
}

std::string HanziEncoder::decode(int code) const {
    return (code >= 0 && code < static_cast<int>(idToHanzi.size())) ? idToHanzi[code] : "";
}

int HanziEncoder::size() const {
    return static_cast<int>(idToHanzi.size());
}