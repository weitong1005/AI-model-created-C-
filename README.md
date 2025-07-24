---

# AI3700: Hanzi Generation Model Based on RNN

AI3700 is a lightweight character-level language model implemented entirely in **C++**, designed for **Chinese character sequence modeling and generation**. It provides an interactive command-line interface and showcases how to build a trainable neural network system from scratch, including matrix operations, forward and backward propagation, optimizer design, and model persistence.

---

## ✨ Motivation & Creation

### Background

This project started as a challenge:
**Can we build a complete neural language model using only pure C++?**
Without relying on deep learning frameworks like PyTorch or TensorFlow, AI3700 implements everything from the ground up using self-written matrix algebra and a recurrent neural network (RNN) for character modeling.

### Architecture

To keep the system simple yet expressive, a **character-level Recurrent Neural Network (RNN)** was adopted. The model learns to generate sequences by learning conditional probabilities between Chinese characters. Each character is represented as a one-hot encoded vector.

### Core Components

* **HanziEncoder**: Loads and maps Chinese characters from `hanzi.txt` to unique IDs
* **Matrix**: A custom matrix class supporting all operations and activations
* **RNNModel**: Core neural network supporting training, inference, and optimization
* **StartModel**: High-level interface with training, response generation, beam search
* **main.cpp**: CLI entry point and model lifecycle control

---

## 📐 Mathematical Foundation

AI3700 implements a standard RNN architecture (non-gated). The model performs the following computations during forward pass and training:

### 1. **Input to Hidden Computation**:

At time step \$t\$, input \$x\_t\$ (one-hot vector), hidden state \$h\_t\$:

$$
a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h
$$

$$
h_t = \phi(a_t)
$$

Where:

* \$\phi\$ is the activation function (Tanh, ReLU, or LeakyReLU)
* \$W\_{xh}\$: input-to-hidden weights
* \$W\_{hh}\$: hidden-to-hidden recurrent weights
* \$b\_h\$: hidden bias

### 2. **Output Layer**:

$$
y_t = W_{hy} h_t + b_y
$$

The prediction distribution is computed via softmax:

$$
\hat{y}_t = \text{softmax}(y_t)
$$

### 3. **Loss Function (Cross-Entropy)**:

$$
L = -\sum_{i=1}^V t_i \log(\hat{y}_i)
$$

Where \$V\$ is the vocabulary size, and \$t\$ is the true one-hot target vector.

### 4. **Optimizer: Adam Update Rule**

Parameters are updated using Adam:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

All matrix operations, NaN protection, gradient clipping, and checkpointing are implemented manually in C++.

---

## 🧪 Engineering & Debugging Strategies

* **Xavier initialization** to prevent gradient explosion
* **Gradient clipping** and **NaN checking** for stability
* **Learning rate decay** during training
* **Beam search** and **temperature sampling** for controlled generation
* Full **UTF-8 character handling** for Chinese compatibility

---

## 📚 Sample Output Flow

```
📖 Loading character dictionary...
✅ Loaded 3700 characters
⏳ First run: training model...
💾 Saving model...
✅ Model saved to model/model.bin

💬 Ready. Type your input (q to quit)
You: 花开富贵
AI: 花开富贵吉祥如意
```

---

## 📁 Project Structure

```
AI3700/
├── CMakeLists.txt         # Build configuration
├── data/
│   └── hanzi.txt          # Pinyin-to-Hanzi dictionary
├── model/                 # Model output directory (auto-created)
├── main.cpp               # Main entry point
└── src/                   # Source code
    ├── file_utils.*       # File I/O helpers
    ├── hanzi_encoder.*    # Hanzi encoding utilities
    ├── matrix.*           # Matrix operations and activation functions
    ├── rnn_model.*        # Core RNN model
    ├── start_model.*      # High-level interface for training/inference
```

---

## 🧠 Features

* Custom encoder supporting 3700+ Chinese characters
* Multi-layer RNN structure with customizable activation functions
* Built-in Adam optimizer with gradient clipping
* Beam search & temperature control for generation diversity
* Support for saving, loading, and incremental training
* Interactive CLI for multi-turn response generation

---

## ⚙️ Build Instructions

### 📌 Dependencies

* C++11 compliant compiler (GCC, Clang, MSVC)
* CMake ≥ 3.10

### 🔨 Compile

```bash
git clone https://github.com/yourname/AI3700.git
cd AI3700
mkdir build && cd build
cmake ..
make
```

After building, the binary `AI3700` will be created and `hanzi.txt` will be copied to the output directory automatically.

---

## 🚀 Usage

Run the model:

```bash
./AI3700
```

On the first run, the model is trained and saved to `model/model.bin`. Subsequent runs will load this model and begin interactive mode.

---

## 🔧 Hyperparameters

You can adjust the following parameters in `main.cpp`:

| Parameter     | Description                               | Default           |
| ------------- | ----------------------------------------- | ----------------- |
| hiddenSize    | Hidden neurons per layer                  | 1 (suggest ≥64)   |
| layers        | Number of RNN layers                      | 1000 (suggest ≤4) |
| contextWindow | Context window size                       | 3                 |
| iterations    | Initial training iterations               | 20000             |
| temperature   | Sampling diversity (higher = more random) | 0.5               |

---

## 📦 Data Format

* `data/hanzi.txt`: Pinyin-to-hanzi mapping (e.g., `ba: 八巴扒吧...`)
* Each character is encoded as a one-hot vector for training.

---

## 🛠 Technical Highlights

* **Matrix class**: Full matrix algebra implementation with activations
* **Model architecture**: Multi-layer vanilla RNN with recurrent weights
* **Optimizer**: In-house Adam with bias correction
* **Training stability**: Gradient clipping, NaN handling, learning rate decay
* **Generation**:

    * Default: Temperature sampling with context window
    * Advanced: Beam search for quality enhancement

---

## 💡 Strengths

* Ultra-lightweight design, hidden size can be as small as 1
* Automatically loads or trains model as needed
* Deterministic generation supported (set `temperature = 0`)
* Multi-turn conversation control with simple CLI

---

## ⚠️ Notes

* First-time training duration depends on `hiddenSize` and `iterations`
* Default model path: `model/model.bin`, customizable in code
* For production use, consider extending with LSTM/GRU/Transformer

---

## 📄 License

This project is licensed under the **W&T**. Contributions and forks are welcome.

---
---

# AI3700：基于 RNN 的汉字生成模型

AI3700 是一个使用 **C++ 原生代码**开发的轻量级字符级语言模型，专注于**汉字序列建模与生成任务**，并提供控制台交互式体验。该项目展示了如何从零开始构建一个具备学习能力的神经网络系统，包括矩阵运算、前向传播、反向传播、优化器设计与模型持久化。

---

## ✨ 项目起源与创造过程

### 背景

本项目的动机来源于一个挑战：\*\*能否使用纯 C++ 构建一个从字符编码到语言生成的神经网络系统？\*\*目标是实现一个不依赖深度学习框架（如 PyTorch、TensorFlow）的模型，通过完整自研的矩阵库和 RNN 结构来模拟语言建模过程。

### 架构选择

为降低复杂性、提升控制力，我们采用了 **字符级 RNN（Recurrent Neural Network）**，模型通过学习汉字之间的条件概率分布进行序列生成。模型的输入为汉字编码后的 one-hot 向量。

### 关键模块

* **HanziEncoder**：从 `hanzi.txt` 加载汉字表，为每个汉字分配唯一编码
* **Matrix 类**：自定义矩阵实现，支持加减乘除、转置和激活函数
* **RNNModel**：核心循环神经网络，具备前向传播、反向传播和训练能力
* **StartModel**：高层封装，提供训练、生成响应、beam search 接口
* **main.cpp**：模型生命周期控制、交互式 CLI 入口

---

## 📐 数学与核心公式

模型采用标准的 RNN 结构（不含门控），基本的前向传播和训练过程包括以下公式：

### 1. **输入层至隐藏层计算**：

对于第 $t$ 个时间步，输入为 one-hot 向量 $x_t$，隐藏状态为 $h_t$：

$$
a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h
$$

$$
h_t = \phi(a_t)
$$

其中：

* $\phi$ 为激活函数（支持 Tanh, ReLU, LeakyReLU）
* $W_{xh}$：输入到隐藏层权重矩阵
* $W_{hh}$：隐藏到隐藏的循环权重
* $b_h$：偏置向量

### 2. **输出层计算**：

$$
y_t = W_{hy} h_t + b_y
$$

经过 Softmax 得到下一个字符的预测分布：

$$
\hat{y}_t = \text{softmax}(y_t)
$$

### 3. **损失函数**（交叉熵）：

$$
L = -\sum_{i=1}^V t_i \log(\hat{y}_i)
$$

其中 $V$ 是词汇表大小，$t$ 是真实 one-hot 标签。

### 4. **优化器**：使用 **Adam** 进行权重更新，参数更新公式：

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

所有矩阵乘法、梯度裁剪、NaN 保护、参数持久化等操作均在 C++ 中手动实现，体现了系统工程能力与数学建模能力的结合。

---

## 🚧 项目创建与调试策略

* 使用 **Xavier 初始化** 避免梯度爆炸
* 实现 **梯度裁剪** 与 **NaN 检查机制**
* 自动衰减学习率以增强训练稳定性
* 提供 `beam search` 与 `温度采样`，提高生成多样性与可控性
* 使用 UTF-8 字符串处理实现中文兼容性

---

## 📚 示例输出流程

```
📖 加载汉字库...
✅ 加载 3700 个汉字
⏳ 首次运行，开始训练模型...
💾 保存模型...
✅ 模型保存至 model/model.bin

💬 模型就绪，输入q退出
你: 花开富贵
AI: 花开富贵吉祥如意
```

---

## 📁 项目结构

```
AI3700/
├── CMakeLists.txt         # 构建配置
├── data/
│   └── hanzi.txt          # 汉字词表（拼音:汉字）
├── model/                 # 模型文件保存路径（程序运行时自动生成）
├── main.cpp               # 主程序入口
└── src/                   # 源码目录
    ├── file_utils.*       # 数据文件读写工具
    ├── hanzi_encoder.*    # 汉字编码器
    ├── matrix.*           # 矩阵实现与激活函数
    ├── rnn_model.*        # RNN 模型主体
    ├── start_model.*      # 高层模型封装与交互接口
```

---

## 🧠 功能概述

* 自定义汉字编码器，支持3700+常用字
* 多层 RNN 网络结构，支持 Tanh/ReLU/LeakyReLU 等激活函数
* 内置 Adam 优化器与梯度裁剪机制
* Beam Search 和温度控制提升生成质量与多样性
* 支持模型保存、加载、自学习、自恢复等机制
* 控制台交互式对话，支持多轮输入响应

---

## ⚙️ 构建方式

### 📌 依赖要求

* C++11 编译器（GCC / Clang / MSVC）
* CMake ≥ 3.10

### 🔨 编译命令

```bash
git clone https://github.com/yourname/AI3700.git
cd AI3700
mkdir build && cd build
cmake ..
make
```

构建完成后将生成 `AI3700` 可执行文件，同时自动将 `hanzi.txt` 拷贝至运行目录。

---

## 🚀 使用方式

在终端中运行程序：

```bash
./AI3700
```

第一次运行将自动训练模型并保存为 `model/model.bin`，此过程可能耗时较长。训练完成后进入交互模式。

---

## 🔧 参数设置

可在 `main.cpp` 中修改如下超参数控制行为：

| 参数            | 描述             | 默认值        |
| ------------- | -------------- | ---------- |
| hiddenSize    | 每层隐藏神经元数       | 1（建议≥64）   |
| layers        | 网络层数           | 1000（建议≤4） |
| contextWindow | 上下文窗口大小        | 3          |
| iterations    | 初始训练迭代次数       | 20000      |
| temperature   | 控制生成多样性（越高越随机） | 0.5        |

---

## 📦 数据说明

* `data/hanzi.txt`：包含拼音和对应汉字列表（如 `ba: 八巴扒吧...`）
* 模型训练基于该字表构建的独热编码（one-hot）向量输入

---

## 🛠 技术细节

* **矩阵实现**：自定义 `Matrix` 类实现矩阵运算与激活函数
* **模型结构**：支持多层RNN（Whh）堆叠，含隐藏→隐藏、输入→隐藏、隐藏→输出权重
* **优化器**：内置 Adam 参数更新逻辑（含偏差修正）
* **训练稳定性**：梯度裁剪、防NaN、权重初始化（Xavier）、自动学习率衰减
* **生成策略**：

    * 默认：温度采样+上下文窗口限制
    * 高级：Beam Search（可调光束宽度）

---

## 💡 亮点与优化

* 极致轻量设计，隐藏层最小可配置为1
* 自动判断模型存在性并加载或训练
* 支持非随机响应（温度=0，禁用随机）
* 多轮响应控制与简洁控制台体验

---

## 📌 注意事项

* 模型初次训练时间视隐藏层大小与迭代次数而定
* 模型文件路径默认保存至 `model/model.bin`，可更改为自定义路径
* 项目为教育/实验用途，如需工业部署建议结合更强的模型（如LSTM/GRU）

---

## 🧾 许可证

本项目B&M 开源发布，欢迎学习、使用与改进。

---
