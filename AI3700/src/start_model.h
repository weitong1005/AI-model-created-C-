#ifndef AI3700_START_MODEL_H
#define AI3700_START_MODEL_H
#include "rnn_model.h"
#include "hanzi_encoder.h"
#include <string>
#include <vector>

class StartModel {
public:
    StartModel(HanziEncoder& encoder, int hiddenSize = 128, int layers = 2);
    bool loadModel(const std::string& path);
    bool saveModel(const std::string& path);
    void train(int iterations = 50000, double lr = 0.001);
    std::string generateResponse(const std::string& input, int length = 20,
                                 double temperature = 0.7, bool random = true);

    // 设置模型参数
    void setContextWindow(int window) { contextWindow = window; }
    void setActivation(RNNModel::ActivationType type) { model.setActivation(type); }

private:
    HanziEncoder& encoder;
    RNNModel model;
    int contextWindow; // 上下文窗口大小

    std::vector<int> strToSeq(const std::string& input);
    std::string seqToStr(const std::vector<int>& seq);

    // 使用beam search提高生成质量
    std::vector<int> beamSearch(const std::vector<int>& inputSeq, int length,
                                int beamWidth = 5, double temperature = 1.0);
};

#endif //AI3700_START_MODEL_H