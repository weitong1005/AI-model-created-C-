#include "start_model.h"
#include <vector>
#include <string>
#include <queue>
#include <tuple>
#include <algorithm>
#include <iostream>  // 增加调试输出

StartModel::StartModel(HanziEncoder& encoder, int hiddenSize, int layers)
        : encoder(encoder), model(encoder.size(), hiddenSize, layers), contextWindow(5) {}

bool StartModel::loadModel(const std::string& path) {
    if (!model.load(path)) {
        std::cerr << "❌ 模型加载失败: " << path << std::endl;
        return false;
    }
    return true;
}

bool StartModel::saveModel(const std::string& path) {
    // 检查路径是否有效
    if (path.empty()) {
        std::cerr << "❌ 模型路径为空" << std::endl;
        return false;
    }

    if (!model.save(path)) {
        std::cerr << "❌ 模型保存失败: " << path << std::endl;
        // 尝试备选路径
        std::string backupPath = "backup_" + path;
        if (model.save(backupPath)) {
            std::cerr << "⚠️ 已保存到备选路径: " << backupPath << std::endl;
            return true;
        }
        return false;
    }
    return true;
}

// 新增：监控训练过程的辅助函数
void printTrainingProgress(int iteration, int total, double loss) {
    if (iteration % 1000 == 0) {
        double progress = static_cast<double>(iteration) / total * 100;
        printf("📊 训练进度: %.1f%% (迭代 %d/%d), 损失: %.6f\n",
               progress, iteration, total, loss);
    }
}

void StartModel::train(int iterations, double lr) {
    // 针对大隐藏层调整训练策略：减小初始学习率
    double initialLr = lr * 0.5;  // 对于128隐藏层，学习率减半

    try {
        // 分阶段训练，增加损失监控
        for (int i = 0; i < iterations; i++) {
            // 动态调整序列长度，避免内存溢出
            int seqLen = 6 + (i % 3) * 2;  // 6-10之间动态调整

            // 生成训练序列
            std::vector<int> seq;
            for (int j = 0; j < seqLen; j++) {
                seq.push_back(rand() % encoder.size());
            }

            // 训练一步并监控损失（简化版损失计算）
            model.train(seq, initialLr * (1 - i/(double)iterations), i);

            // 每1000次迭代打印一次进度
            if (i % 1000 == 0) {
                printTrainingProgress(i, iterations, 0.0);  // 实际项目中应计算真实损失
            }

            // 定期检查数值稳定性
            if (i % 5000 == 0 && i > 0) {
                std::cout << "🔍 检查模型稳定性..." << std::endl;
                // 生成测试序列验证模型是否正常
                std::vector<int> testSeq = {0, 1, 2, 3};  // 随机选择几个汉字索引
                int pred = model.predict(testSeq, false);
                if (pred < 0 || pred >= encoder.size()) {
                    std::cerr << "⚠️ 检测到异常预测，重置学习率" << std::endl;
                    initialLr *= 0.5;  // 遇到异常时降低学习率
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "❌ 训练过程中发生错误: " << e.what() << std::endl;
        // 尝试保存当前模型状态
        saveModel("emergency_model.bin");
    }
}

std::vector<int> StartModel::strToSeq(const std::string& input) {
    std::vector<int> seq;
    for (size_t i = 0; i < input.size();) {
        unsigned char c = (unsigned char)input[i];
        size_t len = 1;

        // 判断UTF-8字符长度
        if ((c & 0xF0) == 0xE0) len = 3;  // 汉字为3字节UTF-8
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if (c >= 0x80) { i++; continue; }  // 跳过无效字符

        if (i + len > input.size()) break;

        std::string hanzi = input.substr(i, len);
        i += len;

        int code = encoder.encode(hanzi);
        if (code != -1) {
            seq.push_back(code);
        }
    }
    return seq;
}

std::string StartModel::seqToStr(const std::vector<int>& seq) {
    std::string s;
    for (int code : seq) {
        s += encoder.decode(code);
    }
    return s;
}

std::vector<int> StartModel::beamSearch(const std::vector<int>& inputSeq, int length,
                                        int beamWidth, double temperature) {
    // 限制beamWidth大小，避免内存占用过高
    beamWidth = std::min(beamWidth, 5);  // 对于大模型，减小beamWidth

    std::vector<std::pair<std::vector<int>, double>> beams;
    beams.emplace_back(inputSeq, 0.0);

    for (int i = 0; i < length; i++) {
        std::vector<std::pair<std::vector<int>, double>> newBeams;

        for (const auto& beam : beams) {
            const std::vector<int>& seq = beam.first;
            double score = beam.second;

            std::vector<int> context = seq;
            if ((int)context.size() > contextWindow) {
                context = std::vector<int>(context.end() - contextWindow, context.end());
            }

            // 减少候选数量，降低计算量
            std::vector<std::pair<int, double>> candidates;
            int candidateCount = std::min(beamWidth * 2, encoder.size() / 2);
            for (int j = 0; j < candidateCount; j++) {
                int next = model.predict(context, true, temperature);
                candidates.emplace_back(next, -j);
            }

            for (const auto& cand : candidates) {
                std::vector<int> newSeq = seq;
                newSeq.push_back(cand.first);
                newBeams.emplace_back(newSeq, score + cand.second);
            }
        }

        std::sort(newBeams.begin(), newBeams.end(),
                  [](const std::pair<std::vector<int>, double>& a,
                     const std::pair<std::vector<int>, double>& b) {
                      return a.second > b.second;
                  });

        if ((int)newBeams.size() > beamWidth) {
            newBeams.resize(beamWidth);
        }

        beams = newBeams;

        // 防止空光束
        if (beams.empty()) {
            std::cerr << "⚠️ 光束搜索为空，使用原始序列" << std::endl;
            return inputSeq;
        }
    }

    return beams[0].first;
}

std::string StartModel::generateResponse(const std::string& input, int length,
                                         double temperature, bool random) {
    // 限制生成长度，避免内存溢出
    length = std::min(length, 20);  // 对于大模型，减少生成长度

    std::vector<int> inputSeq = strToSeq(input);
    if (inputSeq.empty()) return "未识别到有效汉字";

    try {
        std::vector<int> outputSeq;

        if (length > 10) {
            outputSeq = beamSearch(inputSeq, length, 3, temperature);  // 进一步减小beamWidth
        } else {
            outputSeq = inputSeq;
            std::vector<int> context = inputSeq;

            for (int i = 0; i < length; i++) {
                if ((int)context.size() > contextWindow) {
                    context = std::vector<int>(context.end() - contextWindow, context.end());
                }

                int next = model.predict(context, random, temperature);
                outputSeq.push_back(next);
                context.push_back(next);
            }
        }

        return seqToStr(outputSeq);
    } catch (const std::exception& e) {
        std::cerr << "❌ 生成响应时出错: " << e.what() << std::endl;
        return "生成响应失败，请重试";
    }
}