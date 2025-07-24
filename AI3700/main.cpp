#include "hanzi_encoder.h"
#include "file_utils.h"
#include "start_model.h"
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <string>
#include <chrono>

#ifdef _WIN32
#include <direct.h>
#else
#include <unistd.h>
#endif

bool createDirectory(const std::string& path) {
#ifdef _WIN32
    return _mkdir(path.c_str()) == 0;
#else
    mode_t mode = 0755;
    return mkdir(path.c_str(), mode) == 0;
#endif
}

bool fileExists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

int main() {
    auto startTime = std::chrono::high_resolution_clock::now();

    HanziEncoder encoder;
    std::cout << "📖 加载汉字库...\n";
    if (!encoder.loadFromFile("hanzi.txt")) {
        std::cerr << "❌ 汉字库加载失败，请确保hanzi.txt在当前目录\n";
        return 1;
    }
    std::cout << "✅ 加载 " << encoder.size() << " 个汉字\n";

    std::string modelDir = "model";
    if (!fileExists(modelDir) && !createDirectory(modelDir)) {
        std::cerr << "❌ 无法创建模型目录: " << modelDir << "\n";
        modelDir = ".";
        std::cout << "⚠️ 改用当前目录存储模型\n";
    }


    try {
        // 递减隐藏层大小至8，层数保持2
        int hiddenSize = 1;
        int layers = 1000;
        StartModel model(encoder, hiddenSize, layers);
        model.setContextWindow(3);  // 进一步减小上下文窗口
        model.setActivation(RNNModel::TANH);  // Tanh更稳定，适合小模型

        std::string modelPath = modelDir + "/model.bin";
        if (!fileExists(modelPath)) {
            std::cout << "\n⏳ 首次运行，开始训练模型...\n";
            std::cout << "💡 模型配置: 隐藏层大小=" << hiddenSize << ", 层数=" << layers << "\n";

            // 小模型可以适当增加迭代次数
            int iterations = 20000;
            model.train(iterations, 0.0001);

            std::cout << "💾 保存模型...\n";
            if (!model.saveModel(modelPath)) {
                std::cerr << "❌ 模型保存失败！\n";
                return 1;
            } else {
                std::cout << "✅ 模型保存至 " << modelPath << "\n";
            }
        } else {
            std::cout << "\n📂 加载已有模型...\n";
            if (!model.loadModel(modelPath)) {
                std::cerr << "❌ 模型加载失败，尝试重新训练...\n";
                model.train(5000, 0.0001);
                model.saveModel(modelPath);
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;
        std::cout << "\n⏱️  模型准备时间: " << elapsed.count() << " 秒\n";

        std::cout << "\n💬 模型就绪，输入q退出\n";
        std::string input;
        while (true) {
            std::cout << "你: ";
            std::getline(std::cin, input);
            if (input == "q") break;

            std::string response = model.generateResponse(input, 8, 0.5, true);
            std::cout << "AI: " << response << "\n\n";
        }
    } catch (const std::bad_alloc& e) {
        std::cerr << "\n❌ 内存分配失败！尝试更小的隐藏层（4）\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ 程序错误: " << e.what() << "\n";
        return 1;
    }

    return 0;
}