#ifndef AI3700_HANZI_ENCODER_H
#define AI3700_HANZI_ENCODER_H

#include <string>
#include <unordered_map>
#include <vector>

class HanziEncoder {
public:
    bool loadFromFile(const std::string& filename);
    int encode(const std::string& hanzi) const;
    std::string decode(int code) const;
    int size() const;
    void clear();

private:
    std::unordered_map<std::string, int> hanziToId;
    std::vector<std::string> idToHanzi;
};

#endif