---

# AI3700: Hanzi Generation Model Based on RNN

**AI3700** is a lightweight character-level language model implemented entirely in **C++**, designed for **Chinese character sequence modeling and generation**. It provides an interactive command-line interface and showcases how to build a trainable neural network system from scratch, including matrix operations, forward and backward propagation, optimizer design, and model persistence.

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
