#ifndef AI3700_RNN_MODEL_H
#define AI3700_RNN_MODEL_H
#include "matrix.h"
#include <vector>
#include <string>
#include <random>

// Adam优化器参数结构体
struct AdamParams {
    Matrix m;  // 一阶矩估计
    Matrix v;  // 二阶矩估计
    double beta1;
    double beta2;
    double epsilon;

    AdamParams() : beta1(0.9), beta2(0.999), epsilon(1e-8) {}
};

class RNNModel {
public:
    RNNModel(int vocabSize, int hiddenSize, int layers = 2);
    void train(const std::vector<int>& sequence, double learningRate, int epoch);
    int predict(const std::vector<int>& inputSeq, bool random = false, double temperature = 1.0);
    bool save(const std::string& filename);
    bool load(const std::string& filename);
    void selfLearn(int iterations, int seqLen, double lr);

    // 设置激活函数类型
    enum ActivationType { TANH, RELU, LEAKY_RELU, SIGMOID };
    void setActivation(ActivationType type) { activationType = type; }

private:
    int vocabSize;  // 汉字总数
    int hiddenSize; // 隐藏层大小
    int layers;     // 层数
    ActivationType activationType; // 激活函数类型

    // 权重矩阵
    Matrix Wxh;       // 输入→隐藏 [hiddenSize x vocabSize]
    std::vector<Matrix> Whh;  // 隐藏→隐藏 [hiddenSize x hiddenSize]
    Matrix Why;       // 隐藏→输出 [vocabSize x hiddenSize]
    std::vector<Matrix> bh;   // 隐藏层偏置 [hiddenSize x 1]
    Matrix by;        // 输出层偏置 [vocabSize x 1]

    // Adam优化器参数
    AdamParams optWxh, optWhy;
    std::vector<AdamParams> optWhh, optBh;
    AdamParams optBy;

    // 新增：Xavier初始化方法
    void xavierInit(Matrix& m, int fanIn, int fanOut);
    // 新增：梯度裁剪
    void clipGradient(Matrix& grad, double maxNorm);
    // 新增：检查NaN值
    void checkNaN(const Matrix& m, const std::string& msg);

    struct ForwardCache {
        std::vector<Matrix> h;  // 各层隐藏状态 [hiddenSize x 1]
        std::vector<Matrix> a;  // 各层激活前的值 [hiddenSize x 1]
        Matrix y;               // 输出 [vocabSize x 1]

        ForwardCache(int layers, int hiddenSize, int vocabSize) {
            h.resize(layers, Matrix(hiddenSize, 1));
            a.resize(layers, Matrix(hiddenSize, 1));
            y = Matrix(vocabSize, 1);
        }
    };

    ForwardCache forward(const std::vector<int>& inputs);
    Matrix activate(const Matrix& m);
    Matrix activateDerivative(const Matrix& m);

    // Adam优化更新
    void adamUpdate(Matrix& param, AdamParams& opt, const Matrix& grad, double lr, int t);
};

#endif //AI3700_RNN_MODEL_H

////
//// Created by bm on 25-7-23.
////
//
//#ifndef AI3700_RNN_MODEL_H
//#define AI3700_RNN_MODEL_H
//
//#include "matrix.h"
//#include <vector>
//#include <string>
//
//class RNNModel {
//public:
//    RNNModel(int vocabSize, int hiddenSize, int layers = 2);
//
//    void train(const std::vector<int>& sequence, double learningRate);
//    int predict(const std::vector<int>& inputSeq, bool random = false);
//    bool save(const std::string& filename);
//    bool load(const std::string& filename);
//    void selfLearn(int iterations, int seqLen, double lr);
//
//private:
//    int vocabSize;  // 汉字总数
//    int hiddenSize; // 隐藏层大小
//    int layers;     // 层数
//
//    // 权重矩阵定义：
//    // - 仅第一层使用Wxh（输入→隐藏）
//    // - 所有层都使用Whh（隐藏→隐藏）
//    Matrix Wxh;       // 第一层：输入→隐藏 [hiddenSize x vocabSize]
//    std::vector<Matrix> Whh;  // 所有层：隐藏→隐藏 [hiddenSize x hiddenSize]
//    Matrix Why;       // 输出层：隐藏→输出 [vocabSize x hiddenSize]
//    std::vector<Matrix> bh;   // 所有层：隐藏层偏置 [hiddenSize x 1]
//    Matrix by;        // 输出层：偏置 [vocabSize x 1]
//
//    struct ForwardCache {
//        std::vector<Matrix> h;  // 各层隐藏状态 [hiddenSize x 1]
//        Matrix y;               // 输出 [vocabSize x 1]
//        ForwardCache(int layers, int hiddenSize, int vocabSize) {
//            h.resize(layers, Matrix(hiddenSize, 1));
//            y = Matrix(vocabSize, 1);
//        }
//    };
//
//    ForwardCache forward(const std::vector<int>& inputs);
//};
//
//
//#endif //AI3700_RNN_MODEL_H
