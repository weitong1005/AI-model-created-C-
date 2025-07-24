#ifndef AI3700_FILE_UTILS_H
#define AI3700_FILE_UTILS_H

#include "hanzi_encoder.h"
#include <string>

class FileUtils {
public:
    static bool generateDataFiles(const HanziEncoder& encoder,
                                  const std::string& dataFile = "hanzi_data.bin",
                                  const std::string& codeFile = "hanzi_codes.bin");

    static bool loadFromDataFiles(HanziEncoder& encoder,
                                  const std::string& dataFile = "hanzi_data.bin",
                                  const std::string& codeFile = "hanzi_codes.bin");
};

#endif