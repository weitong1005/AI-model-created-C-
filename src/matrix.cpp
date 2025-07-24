#include "matrix.h"
#include <cstdlib>
#include <ctime>

Matrix::Matrix(int rows, int cols, bool random) : rows(rows), cols(cols) {
    data.resize(rows, std::vector<double>(cols, 0.0));
    if (random) {
        std::srand(std::time(nullptr));
        randomize();
    }
}

Matrix::Matrix(const std::vector<std::vector<double>>& initData)
        : data(initData), rows(initData.size()), cols(initData[0].size()) {}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        std::cerr << "矩阵加法维度不匹配: (" << rows << "x" << cols
                  << ") + (" << other.rows << "x" << other.cols << ")\n";
        return Matrix();
    }

    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result.data[i][j] = data[i][j] + other.data[i][j];
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        std::cerr << "矩阵减法维度不匹配: (" << rows << "x" << cols
                  << ") - (" << other.rows << "x" << other.cols << ")\n";
        return Matrix();
    }

    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result.data[i][j] = data[i][j] - other.data[i][j];
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        std::cerr << "致命错误：矩阵乘法维度不匹配 (" << rows << "x" << cols
                  << ") * (" << other.rows << "x" << other.cols << ")\n";
        std::exit(1);
    }

    Matrix result(rows, other.cols);
    for (int i = 0; i < rows; i++) {
        for (int k = 0; k < cols; k++) {
            if (data[i][k] == 0) continue;
            for (int j = 0; j < other.cols; j++) {
                result.data[i][j] += data[i][k] * other.data[k][j];
            }
        }
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result.data[i][j] = data[i][j] * scalar;
    return result;
}

Matrix Matrix::operator/(double scalar) const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result.data[i][j] = data[i][j] / scalar;
    return result;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    if (rows != other.rows || cols != other.cols) {
        std::cerr << "矩阵加法维度不匹配\n";
        return *this;
    }

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            data[i][j] += other.data[i][j];
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (rows != other.rows || cols != other.cols) {
        std::cerr << "矩阵减法维度不匹配\n";
        return *this;
    }

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            data[i][j] -= other.data[i][j];
    return *this;
}

Matrix& Matrix::operator*=(double scalar) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            data[i][j] *= scalar;
    return *this;
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[j][i] = data[i][j];
        }
    }
    return result;
}

Matrix Matrix::sigmoid(const Matrix& m) {
    Matrix result(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++)
            result.data[i][j] = 1.0 / (1.0 + exp(-m.data[i][j]));
    return result;
}

Matrix Matrix::sigmoidDerivative(const Matrix& m) {
    Matrix s = sigmoid(m);
    Matrix result(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++)
            result.data[i][j] = s.data[i][j] * (1 - s.data[i][j]);
    return result;
}

Matrix Matrix::tanh(const Matrix& m) {
    Matrix result(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++)
            result.data[i][j] = std::tanh(m.data[i][j]);
    return result;
}

Matrix Matrix::tanhDerivative(const Matrix& m) {
    Matrix result(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++) {
            double t = std::tanh(m.data[i][j]);
            result.data[i][j] = 1 - t * t;
        }
    return result;
}

Matrix Matrix::relu(const Matrix& m) {
    Matrix result(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++)
            result.data[i][j] = std::max(0.0, m.data[i][j]);
    return result;
}

Matrix Matrix::reluDerivative(const Matrix& m) {
    Matrix result(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++)
            result.data[i][j] = (m.data[i][j] > 0) ? 1.0 : 0.0;
    return result;
}

Matrix Matrix::leakyRelu(const Matrix& m, double alpha) {
    Matrix result(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++)
            result.data[i][j] = std::max(alpha * m.data[i][j], m.data[i][j]);
    return result;
}

Matrix Matrix::leakyReluDerivative(const Matrix& m, double alpha) {
    Matrix result(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++)
            result.data[i][j] = (m.data[i][j] > 0) ? 1.0 : alpha;
    return result;
}

void Matrix::randomize(double min, double max) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            data[i][j] = min + (max - min) * ((double)rand() / RAND_MAX);
}

void Matrix::zeros() {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            data[i][j] = 0.0;
}

void Matrix::ones() {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            data[i][j] = 1.0;
}

////
//// Created by bm on 25-7-23.
////
//#include "matrix.h"
//#include <cstdlib>
//#include <cmath>
//
//Matrix::Matrix(int rows, int cols, bool random) : rows(rows), cols(cols) {
//    data.resize(rows, std::vector<double>(cols, 0.0));
//    if (random) randomize();
//}
//
//Matrix::Matrix(const std::vector<std::vector<double>>& initData)
//        : data(initData), rows(initData.size()), cols(initData[0].size()) {}
//
//Matrix Matrix::operator+(const Matrix& other) const {
//    if (rows != other.rows || cols != other.cols) {
//        std::cerr << "矩阵加法维度不匹配\n";
//        return Matrix();
//    }
//    Matrix result(rows, cols);
//    for (int i = 0; i < rows; i++)
//        for (int j = 0; j < cols; j++)
//            result.data[i][j] = data[i][j] + other.data[i][j];
//    return result;
//}
//
//Matrix Matrix::operator-(const Matrix& other) const {
//    if (rows != other.rows || cols != other.cols) {
//        std::cerr << "矩阵减法维度不匹配\n";
//        return Matrix();
//    }
//    Matrix result(rows, cols);
//    for (int i = 0; i < rows; i++)
//        for (int j = 0; j < cols; j++)
//            result.data[i][j] = data[i][j] - other.data[i][j];
//    return result;
//}
//
//Matrix Matrix::operator*(const Matrix& other) const {
//    // 严格检查维度匹配
//    if (cols != other.rows) {
//        std::cerr << "致命错误：矩阵乘法维度不匹配 (" << rows << "x" << cols
//                  << ") * (" << other.rows << "x" << other.cols << ")\n";
//        std::exit(1); // 直接退出，避免后续错误
//    }
//    Matrix result(rows, other.cols);
//    for (int i = 0; i < rows; i++) {
//        for (int k = 0; k < cols; k++) {
//            if (data[i][k] == 0) continue;
//            for (int j = 0; j < other.cols; j++) {
//                result.data[i][j] += data[i][k] * other.data[k][j];
//            }
//        }
//    }
//    return result;
//}
//
//Matrix Matrix::operator*(double scalar) const {
//    Matrix result(rows, cols);
//    for (int i = 0; i < rows; i++)
//        for (int j = 0; j < cols; j++)
//            result.data[i][j] = data[i][j] * scalar;
//    return result;
//}
//
//Matrix Matrix::transpose() const {
//    Matrix result(cols, rows);
//    for (int i = 0; i < rows; i++) {
//        for (int j = 0; j < cols; j++) {
//            result.data[j][i] = data[i][j];
//        }
//    }
////     调试：验证转置后的维度
//     std::cout << "转置前: " << rows << "x" << cols
//               << " → 转置后: " << result.rows << "x" << result.cols << "\n";
//    return result;
//}
//
//Matrix Matrix::sigmoid(const Matrix& m) {
//    Matrix result(m.rows, m.cols);
//    for (int i = 0; i < m.rows; i++)
//        for (int j = 0; j < m.cols; j++)
//            result.data[i][j] = 1.0 / (1.0 + exp(-m.data[i][j]));
//    return result;
//}
//
//Matrix Matrix::tanh(const Matrix& m) {
//    Matrix result(m.rows, m.cols);
//    for (int i = 0; i < m.rows; i++)
//        for (int j = 0; j < m.cols; j++)
//            result.data[i][j] = std::tanh(m.data[i][j]);
//    return result;
//}
//
//Matrix Matrix::tanhDerivative(const Matrix& m) {
//    Matrix result(m.rows, m.cols);
//    for (int i = 0; i < m.rows; i++)
//        for (int j = 0; j < m.cols; j++) {
//            double t = std::tanh(m.data[i][j]);
//            result.data[i][j] = 1 - t * t;
//        }
//    return result;
//}
//
//void Matrix::randomize(double min, double max) {
//    for (int i = 0; i < rows; i++)
//        for (int j = 0; j < cols; j++)
//            data[i][j] = min + (max - min) * ((double)rand() / RAND_MAX);
//}

