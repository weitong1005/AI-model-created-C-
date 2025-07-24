#include "rnn_model.h"
#include <fstream>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

// 添加NaN检查函数
bool isNaN(double x) {
    return x != x;
}

// 添加梯度裁剪函数
void clipGradients(Matrix& grad, double maxNorm) {
    double norm = 0.0;
    for (int i = 0; i < grad.rows; i++) {
        for (int j = 0; j < grad.cols; j++) {
            norm += grad.data[i][j] * grad.data[i][j];
        }
    }
    norm = sqrt(norm);

    if (norm > maxNorm && norm > 0) {
        double scale = maxNorm / norm;
        grad *= scale;
    }
}

RNNModel::RNNModel(int vocabSize, int hiddenSize, int layers)
        : vocabSize(vocabSize), hiddenSize(hiddenSize), layers(layers), activationType(TANH) {
    // 改进权重初始化：使用Xavier初始化，解决初始值过大问题
    double wxhScale = sqrt(1.0 / vocabSize);
    Wxh = Matrix(hiddenSize, vocabSize, true);
    Wxh *= wxhScale;  // 根据输入维度缩放

    Whh.resize(layers);
    double whhScale = sqrt(1.0 / hiddenSize);
    for (int i = 0; i < layers; i++) {
        Whh[i] = Matrix(hiddenSize, hiddenSize, true);
        Whh[i] *= whhScale;  // 根据隐藏层维度缩放
    }

    double whyScale = sqrt(1.0 / hiddenSize);
    Why = Matrix(vocabSize, hiddenSize, true);
    Why *= whyScale;

    // 偏置初始化为较小的常数，避免初始为0
    bh.resize(layers, Matrix(hiddenSize, 1));
    for (int i = 0; i < layers; i++) {
        for (int j = 0; j < hiddenSize; j++) {
            bh[i].data[j][0] = 0.01;  // 小的正值初始化
        }
    }

    by = Matrix(vocabSize, 1);
    for (int i = 0; i < vocabSize; i++) {
        by.data[i][0] = 0.01;
    }

    // 初始化Adam优化器参数
    optWxh.m = Matrix(hiddenSize, vocabSize);
    optWxh.v = Matrix(hiddenSize, vocabSize);
    optWhy.m = Matrix(vocabSize, hiddenSize);
    optWhy.v = Matrix(vocabSize, hiddenSize);

    optWhh.resize(layers);
    optBh.resize(layers);
    for (int i = 0; i < layers; i++) {
        optWhh[i].m = Matrix(hiddenSize, hiddenSize);
        optWhh[i].v = Matrix(hiddenSize, hiddenSize);
        optBh[i].m = Matrix(hiddenSize, 1);
        optBh[i].v = Matrix(hiddenSize, 1);
    }

    optBy.m = Matrix(vocabSize, 1);
    optBy.v = Matrix(vocabSize, 1);
}

Matrix RNNModel::activate(const Matrix& m) {
    // 激活函数输出后检查并处理NaN
    Matrix result = Matrix::leakyRelu(m);  // Leaky ReLU更不容易出现死亡神经元

    // 检查并替换NaN值
    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            if (isNaN(result.data[i][j])) {
                result.data[i][j] = 0.01;  // 用小值替换NaN
            }
        }
    }

    return result;
}

Matrix RNNModel::activateDerivative(const Matrix& m) {
    return Matrix::leakyReluDerivative(m);
}

// 改进的softmax函数，添加数值稳定处理
Matrix softmax(const Matrix& m) {
    Matrix result(m.rows, m.cols);
    double maxVal = m.data[0][0];

    // 找到最大值，用于数值稳定
    for (int i = 0; i < m.rows; i++) {
        if (m.data[i][0] > maxVal) {
            maxVal = m.data[i][0];
        }
    }

    // 计算指数并减去最大值防止溢出
    double sum = 0.0;
    for (int i = 0; i < m.rows; i++) {
        result.data[i][0] = exp(m.data[i][0] - maxVal);
        sum += result.data[i][0];
    }

    // 归一化
    for (int i = 0; i < m.rows; i++) {
        result.data[i][0] /= sum;
        // 处理可能的NaN（当sum为0时）
        if (isNaN(result.data[i][0])) {
            result.data[i][0] = 1.0 / m.rows;
        }
    }

    return result;
}

// 修改forward方法中的矩阵运算部分
RNNModel::ForwardCache RNNModel::forward(const std::vector<int>& inputs) {
    ForwardCache cache(layers, hiddenSize, vocabSize);

    for (size_t t = 0; t < inputs.size(); t++) {
        int idx = inputs[t];
        if (idx < 0 || idx >= vocabSize) {
            std::cerr << "无效汉字索引: " << idx << "\n";
            std::exit(1);
        }

        // 输入层独热编码 [vocabSize x 1]
        Matrix x(vocabSize, 1);
        x.data[idx][0] = 1.0;

        // 第一层计算：修复矩阵维度不匹配
        // [hiddenSize x 1] = ([hiddenSize x vocabSize] * [vocabSize x 1]) +
        //                   ([hiddenSize x hiddenSize] * [hiddenSize x 1]) + [hiddenSize x 1]
        Matrix wxhResult = Wxh * x;          // [hiddenSize x 1]
        Matrix whhResult = Whh[0] * cache.h[0];  // [hiddenSize x 1]
        cache.a[0] = wxhResult + whhResult + bh[0];  // 正确的维度相加
        cache.h[0] = activate(cache.a[0]);

        // 深层计算
        for (int l = 1; l < layers; l++) {
            // [hiddenSize x 1] = [hiddenSize x hiddenSize] * [hiddenSize x 1] + [hiddenSize x 1]
            cache.a[l] = Whh[l] * cache.h[l-1] + bh[l];
            cache.h[l] = activate(cache.a[l]);
        }

        // 输出层计算 [vocabSize x 1] = [vocabSize x hiddenSize] * [hiddenSize x 1] + [vocabSize x 1]
        cache.y = Why * cache.h[layers-1] + by;
    }

    return cache;
}

void RNNModel::adamUpdate(Matrix& param, AdamParams& opt, const Matrix& grad, double lr, int t) {
    // 复制梯度用于修改
    Matrix safeGrad = grad;

    // 检查并替换梯度中的NaN
    for (int i = 0; i < safeGrad.rows; i++) {
        for (int j = 0; j < safeGrad.cols; j++) {
            if (isNaN(safeGrad.data[i][j])) {
                safeGrad.data[i][j] = 0.0;  // NaN梯度替换为0
            }
        }
    }

    // 梯度裁剪，防止梯度爆炸
    clipGradients(safeGrad, 5.0);  // 最大范数设为5.0

    for (int i = 0; i < param.rows; i++) {
        for (int j = 0; j < param.cols; j++) {
            // 更新一阶矩估计
            opt.m.data[i][j] = opt.beta1 * opt.m.data[i][j] + (1 - opt.beta1) * safeGrad.data[i][j];
            // 更新二阶矩估计
            opt.v.data[i][j] = opt.beta2 * opt.v.data[i][j] + (1 - opt.beta2) * safeGrad.data[i][j] * safeGrad.data[i][j];

            // 偏差修正
            double m_hat = opt.m.data[i][j] / (1 - pow(opt.beta1, t));
            double v_hat = opt.v.data[i][j] / (1 - pow(opt.beta2, t));

            // 参数更新，添加额外保护防止NaN
            if (!isNaN(m_hat) && !isNaN(v_hat) && v_hat > 0) {
                param.data[i][j] -= lr * m_hat / (sqrt(v_hat) + opt.epsilon);
            }

            // 最后检查参数是否为NaN
            if (isNaN(param.data[i][j])) {
                param.data[i][j] = 0.01;  // 重置为小值
            }
        }
    }
}

void RNNModel::train(const std::vector<int>& sequence, double learningRate, int epoch) {
    if (sequence.size() < 2) return;

    std::vector<int> inputs(sequence.begin(), sequence.end() - 1);
    std::vector<int> targets(sequence.begin() + 1, sequence.end());

    auto cache = forward(inputs);
    int lastTarget = targets.back();
    if (lastTarget < 0 || lastTarget >= vocabSize) return;

    // 输出误差 [vocabSize x 1]
    Matrix dy(vocabSize, 1);
    for (int i = 0; i < vocabSize; i++) {
        dy.data[i][0] = cache.y.data[i][0] - (i == lastTarget ? 1.0 : 0.0);
    }

    // 计算误差项
    std::vector<Matrix> dh(layers, Matrix(hiddenSize, 1));

    // 最后一层隐藏层误差
    Matrix whyT = Why.transpose();  // [hiddenSize x vocabSize]
    dh[layers-1] = whyT * dy;       // [hiddenSize x 1] = [hiddenSize x vocabSize] * [vocabSize x 1]
    dh[layers-1] = dh[layers-1] * activateDerivative(cache.a[layers-1]);

    // 深层误差反向传播
    for (int l = layers-2; l >= 0; l--) {
        Matrix whhT = Whh[l+1].transpose();  // [hiddenSize x hiddenSize]
        dh[l] = whhT * dh[l+1];              // [hiddenSize x 1] = [hiddenSize x hiddenSize] * [hiddenSize x 1]
        dh[l] = dh[l] * activateDerivative(cache.a[l]);
    }

    // 更新权重（确保所有矩阵乘法维度正确）
    Matrix whyGrad = dy * cache.h[layers-1].transpose();  // [vocabSize x hiddenSize]
    adamUpdate(Why, optWhy, whyGrad, learningRate, epoch);

    // 更新第一层权重
    Matrix x(vocabSize, 1);
    x.data[inputs.back()][0] = 1.0;
    Matrix wxhGrad = dh[0] * x.transpose();  // [hiddenSize x vocabSize]
    adamUpdate(Wxh, optWxh, wxhGrad, learningRate, epoch);

    // 更新隐藏层权重
    for (int l = 0; l < layers; l++) {
        Matrix hPrev = (l == 0) ? cache.h[0] : cache.h[l-1];
        Matrix whhGrad = dh[l] * hPrev.transpose();  // [hiddenSize x hiddenSize]
        adamUpdate(Whh[l], optWhh[l], whhGrad, learningRate, epoch);
        adamUpdate(bh[l], optBh[l], dh[l], learningRate, epoch);
    }

    adamUpdate(by, optBy, dy, learningRate, epoch);
}

int RNNModel::predict(const std::vector<int>& inputSeq, bool random, double temperature) {
    auto cache = forward(inputSeq);

    // 应用温度调整概率分布，使用稳定的softmax
    Matrix probs = softmax(cache.y);

    // 温度调整
    for (int i = 0; i < vocabSize; i++) {
        if (probs.data[i][0] <= 0) {
            probs.data[i][0] = 1e-10;  // 防止log(0)
        }
        probs.data[i][0] = pow(probs.data[i][0], 1.0 / temperature);
    }

    // 归一化概率
    double sum = 0.0;
    for (int i = 0; i < vocabSize; i++) {
        sum += probs.data[i][0];
    }

    // 处理sum为0的情况
    if (sum <= 0) {
        for (int i = 0; i < vocabSize; i++) {
            probs.data[i][0] = 1.0 / vocabSize;
        }
        sum = 1.0;
    }

    for (int i = 0; i < vocabSize; i++) {
        probs.data[i][0] /= sum;
        // 最终检查NaN
        if (isNaN(probs.data[i][0])) {
            probs.data[i][0] = 1.0 / vocabSize;
        }
    }

    if (!random) {
        // 选择概率最大的字符
        int maxIdx = 0;
        double maxVal = probs.data[0][0];
        for (int i = 1; i < vocabSize; i++) {
            if (probs.data[i][0] > maxVal) {
                maxVal = probs.data[i][0];
                maxIdx = i;
            }
        }
        return maxIdx;
    } else {
        // 基于概率分布随机选择
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        double r = dis(gen);

        double accum = 0.0;
        for (int i = 0; i < vocabSize; i++) {
            accum += probs.data[i][0];
            if (accum >= r) {
                return i;
            }
        }
        return vocabSize - 1; // fallback
    }
}

// 保持其他函数不变...
bool RNNModel::save(const std::string& filename) {
    std::ofstream fout(filename, std::ios::binary);
    if (!fout) return false;

    fout.write((char*)&vocabSize, sizeof(vocabSize));
    fout.write((char*)&hiddenSize, sizeof(hiddenSize));
    fout.write((char*)&layers, sizeof(layers));
    fout.write((char*)&activationType, sizeof(activationType));

    auto saveMatrix = [&](const Matrix& m) {
        fout.write((char*)&m.rows, sizeof(m.rows));
        fout.write((char*)&m.cols, sizeof(m.cols));
        for (const auto& row : m.data)
            fout.write((char*)row.data(), row.size() * sizeof(double));
    };

    saveMatrix(Wxh);
    for (const auto& m : Whh) saveMatrix(m);
    saveMatrix(Why);
    for (const auto& m : bh) saveMatrix(m);
    saveMatrix(by);

    return true;
}

bool RNNModel::load(const std::string& filename) {
    std::ifstream fin(filename, std::ios::binary);
    if (!fin) return false;

    fin.read((char*)&vocabSize, sizeof(vocabSize));
    fin.read((char*)&hiddenSize, sizeof(hiddenSize));
    fin.read((char*)&layers, sizeof(layers));

    int actType;
    fin.read((char*)&actType, sizeof(actType));
    activationType = static_cast<ActivationType>(actType);

    auto loadMatrix = [&](Matrix& m) {
        int rows, cols;
        fin.read((char*)&rows, sizeof(rows));
        fin.read((char*)&cols, sizeof(cols));
        m = Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            m.data[i].resize(cols);
            fin.read((char*)m.data[i].data(), cols * sizeof(double));
        }
    };

    loadMatrix(Wxh);
    Whh.resize(layers);
    for (int i = 0; i < layers; i++) loadMatrix(Whh[i]);
    loadMatrix(Why);
    bh.resize(layers);
    for (int i = 0; i < layers; i++) loadMatrix(bh[i]);
    loadMatrix(by);

    // 重新初始化优化器参数
    optWxh = AdamParams();
    optWhy = AdamParams();
    optWhh.resize(layers);
    optBh.resize(layers);
    optBy = AdamParams();

    return true;
}

void RNNModel::selfLearn(int iterations, int seqLen, double lr) {
    double currentLr = lr;
    int consecutiveErrors = 0;

    for (int i = 0; i < iterations; i++) {
        try {
            std::vector<int> seq;
            for (int j = 0; j < seqLen; j++) {
                seq.push_back(rand() % vocabSize);
            }

            // 每1000次迭代降低一次学习率
            if (i % 1000 == 0 && i > 0) {
                currentLr *= 0.95;  // 5%衰减率
            }

            train(seq, currentLr, i);
            consecutiveErrors = 0;  // 重置错误计数

            // 每5000次迭代输出进度
            if (i % 5000 == 0) {
                printf("📊 训练进度: %.1f%% (迭代 %d/%d)\n",
                       (double)i/iterations*100, i, iterations);
            }
        } catch (const std::exception& e) {
            std::cerr << "❌ 训练步骤失败: " << e.what() << "\n";
            consecutiveErrors++;

            // 如果连续出错，降低学习率
            if (consecutiveErrors >= 5) {
                currentLr *= 0.5;
                std::cerr << "⚠️ 连续错误，降低学习率至: " << currentLr << "\n";
                consecutiveErrors = 0;

                // 如果学习率过小，仍然出错，则终止训练
                if (currentLr < 1e-8) {
                    throw std::runtime_error("学习率已过小但仍出错，无法继续训练");
                }
            }
        }
    }
}



////
//// Created by bm on 25-7-23.
////
//#include "rnn_model.h"
//#include <fstream>
//#include <cmath>
//#include <iostream>
//
//RNNModel::RNNModel(int vocabSize, int hiddenSize, int layers)
//        : vocabSize(vocabSize), hiddenSize(hiddenSize), layers(layers) {
//    // 初始化权重矩阵（确保维度正确）
//    Wxh = Matrix(hiddenSize, vocabSize, true);  // 输入→隐藏: [h x v]
//    Whh.resize(layers, Matrix(hiddenSize, hiddenSize, true));  // 隐藏→隐藏: [h x h]
//    Why = Matrix(vocabSize, hiddenSize, true);  // 隐藏→输出: [v x h]
//    bh.resize(layers, Matrix(hiddenSize, 1, true));  // 隐藏层偏置: [h x 1]
//    by = Matrix(vocabSize, 1, true);  // 输出层偏置: [v x 1]
//}
//
//RNNModel::ForwardCache RNNModel::forward(const std::vector<int>& inputs) {
//    ForwardCache cache(layers, hiddenSize, vocabSize);
//
//    for (size_t t = 0; t < inputs.size(); t++) {
//        // 输入层独热编码 [v x 1]
//        int idx = inputs[t];
//        if (idx < 0 || idx >= vocabSize) {
//            std::cerr << "无效汉字索引: " << idx << "\n";
//            std::exit(1);
//        }
//        Matrix x(vocabSize, 1);
//        x.data[idx][0] = 1.0;
//
//        // 第一层计算 [h x 1] = [h x v] * [v x 1] + [h x h] * [h x 1] + [h x 1]
//        Matrix h1 = Wxh * x + Whh[0] * cache.h[0] + bh[0];
//        cache.h[0] = Matrix::tanh(h1);
//
//        // 深层计算（仅1层时不执行）
//        for (int l = 1; l < layers; l++) {
//            Matrix hl = Whh[l] * cache.h[l-1] + bh[l];
//            cache.h[l] = Matrix::tanh(hl);
//        }
//
//        // 输出层计算 [v x 1] = [v x h] * [h x 1] + [v x 1]
//        cache.y = Why * cache.h[layers-1] + by;
//    }
//    return cache;
//}
//
//void RNNModel::train(const std::vector<int>& sequence, double learningRate) {
//    if (sequence.size() < 2) return;
//    std::vector<int> inputs(sequence.begin(), sequence.end() - 1);
//    std::vector<int> targets(sequence.begin() + 1, sequence.end());
//
//    auto cache = forward(inputs);
//    int lastTarget = targets.back();
//    if (lastTarget < 0 || lastTarget >= vocabSize) return;
//
//    // 输出误差 [v x 1]
//    Matrix dy(vocabSize, 1);
//    for (int i = 0; i < vocabSize; i++) {
//        dy.data[i][0] = cache.y.data[i][0] - (i == lastTarget ? 1.0 : 0.0);
//    }
//
//    // 最后一层隐藏层误差 [h x 1] = [h x v] * [v x 1]（Why转置后是[h x v]）
//    std::vector<Matrix> dh(layers);
//    dh[layers-1] = Why.transpose() * dy;
//    dh[layers-1] = dh[layers-1] * Matrix::tanhDerivative(cache.h[layers-1]);
//
//    // 深层误差反向传播（仅1层时不执行）
//    for (int l = layers-2; l >= 0; l--) {
//        dh[l] = Whh[l+1].transpose() * dh[l+1];
//        dh[l] = dh[l] * Matrix::tanhDerivative(cache.h[l]);
//    }
//
//    // 更新输出层权重 [v x h] = [v x h] - [v x 1] * [1 x h] * lr
//    Matrix why_grad = dy * cache.h[layers-1].transpose();  // 关键修复：使用转置
//    Why = Why - why_grad * learningRate;
//    by = by - dy * learningRate;
//
//    // 更新第一层输入权重 [h x v] = [h x v] - [h x 1] * [1 x v] * lr
//    int lastInput = inputs.back();
//    Matrix x(vocabSize, 1);
//    x.data[lastInput][0] = 1.0;
//    Matrix wxh_grad = dh[0] * x.transpose();  // 关键修复：使用转置
//    Wxh = Wxh - wxh_grad * learningRate;
//
//    // 更新隐藏层权重 [h x h] = [h x h] - [h x 1] * [1 x h] * lr
//    for (int l = 0; l < layers; l++) {
//        Matrix h_prev = (l == 0) ? cache.h[0] : cache.h[l-1];
//        Matrix whh_grad = dh[l] * h_prev.transpose();  // 关键修复：使用转置
//        Whh[l] = Whh[l] - whh_grad * learningRate;
//        bh[l] = bh[l] - dh[l] * learningRate;
//    }
//}
//
//int RNNModel::predict(const std::vector<int>& inputSeq, bool random) {
//    auto cache = forward(inputSeq);
//    if (random) {
//        double sum = 0;
//        for (int i = 0; i < vocabSize; i++)
//            sum += exp(cache.y.data[i][0]);
//        double r = (double)rand() / RAND_MAX;
//        double accum = 0;
//        for (int i = 0; i < vocabSize; i++) {
//            accum += exp(cache.y.data[i][0]) / sum;
//            if (accum >= r) return i;
//        }
//    }
//    int maxIdx = 0;
//    double maxVal = cache.y.data[0][0];
//    for (int i = 1; i < vocabSize; i++) {
//        if (cache.y.data[i][0] > maxVal) {
//            maxVal = cache.y.data[i][0];
//            maxIdx = i;
//        }
//    }
//    return maxIdx;
//}
//
//void RNNModel::selfLearn(int iterations, int seqLen, double lr) {
//    for (int i = 0; i < iterations; i++) {
//        std::vector<int> seq;
//        for (int j = 0; j < seqLen; j++)
//            seq.push_back(rand() % vocabSize);
//        train(seq, lr);
//        if (i % 5000 == 0)
//            printf("迭代 %d/%d\n", i, iterations);
//    }
//}
//
//bool RNNModel::save(const std::string& filename) {
//    std::ofstream fout(filename, std::ios::binary);
//    if (!fout) return false;
//
//    fout.write((char*)&vocabSize, sizeof(vocabSize));
//    fout.write((char*)&hiddenSize, sizeof(hiddenSize));
//    fout.write((char*)&layers, sizeof(layers));
//
//    auto saveMatrix = [&](const Matrix& m) {
//        fout.write((char*)&m.rows, sizeof(m.rows));
//        fout.write((char*)&m.cols, sizeof(m.cols));
//        for (const auto& row : m.data)
//            fout.write((char*)row.data(), row.size() * sizeof(double));
//    };
//
//    saveMatrix(Wxh);
//    for (const auto& m : Whh) saveMatrix(m);
//    saveMatrix(Why);
//    for (const auto& m : bh) saveMatrix(m);
//    saveMatrix(by);
//
//    return true;
//}
//
//bool RNNModel::load(const std::string& filename) {
//    std::ifstream fin(filename, std::ios::binary);
//    if (!fin) return false;
//
//    fin.read((char*)&vocabSize, sizeof(vocabSize));
//    fin.read((char*)&hiddenSize, sizeof(hiddenSize));
//    fin.read((char*)&layers, sizeof(layers));
//
//    auto loadMatrix = [&](Matrix& m) {
//        int rows, cols;
//        fin.read((char*)&rows, sizeof(rows));
//        fin.read((char*)&cols, sizeof(cols));
//        m = Matrix(rows, cols);
//        for (int i = 0; i < rows; i++) {
//            m.data[i].resize(cols);
//            fin.read((char*)m.data[i].data(), cols * sizeof(double));
//        }
//    };
//
//    loadMatrix(Wxh);
//    Whh.resize(layers);
//    for (int i = 0; i < layers; i++) loadMatrix(Whh[i]);
//    loadMatrix(Why);
//    bh.resize(layers);
//    for (int i = 0; i < layers; i++) loadMatrix(bh[i]);
//    loadMatrix(by);
//
//    return true;
//}
