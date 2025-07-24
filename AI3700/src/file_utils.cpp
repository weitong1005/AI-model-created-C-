#include "file_utils.h"
#include <fstream>

bool FileUtils::generateDataFiles(const HanziEncoder& encoder,
                                  const std::string& dataFile,
                                  const std::string& codeFile) {
    // 生成数据文件
    std::ofstream dataOut(dataFile, std::ios::binary);
    if (!dataOut) return false;

    int count = encoder.size();
    dataOut.write(reinterpret_cast<const char*>(&count), sizeof(count));

    for (int i = 0; i < count; ++i) {
        std::string hanzi = encoder.decode(i);
        int len = hanzi.size();
        dataOut.write(reinterpret_cast<const char*>(&len), sizeof(len));
        dataOut.write(hanzi.data(), len);
    }
    dataOut.close();

    // 生成编码文件
    std::ofstream codeOut(codeFile, std::ios::binary);
    if (!codeOut) return false;

    codeOut.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (int i = 0; i < count; ++i) {
        std::string hanzi = encoder.decode(i);
        int len = hanzi.size();
        codeOut.write(reinterpret_cast<const char*>(&len), sizeof(len));
        codeOut.write(hanzi.data(), len);
        codeOut.write(reinterpret_cast<const char*>(&i), sizeof(i));
    }
    codeOut.close();

    return true;
}

// loadFromDataFiles 实现保持不变