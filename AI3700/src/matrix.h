#ifndef AI3700_MATRIX_H
#define AI3700_MATRIX_H
#include <vector>
#include <iostream>
#include <cmath>

class Matrix {
public:
    std::vector<std::vector<double>> data;
    int rows;
    int cols;

    Matrix() : rows(0), cols(0) {}
    Matrix(int rows, int cols, bool random = false);
    Matrix(const std::vector<std::vector<double>>& initData);

    // 矩阵运算
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(double scalar) const;
    Matrix operator/(double scalar) const;
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(double scalar);

    Matrix transpose() const;

    // 激活函数
    static Matrix sigmoid(const Matrix& m);
    static Matrix sigmoidDerivative(const Matrix& m);
    static Matrix tanh(const Matrix& m);
    static Matrix tanhDerivative(const Matrix& m);
    static Matrix relu(const Matrix& m);
    static Matrix reluDerivative(const Matrix& m);
    static Matrix leakyRelu(const Matrix& m, double alpha = 0.01);
    static Matrix leakyReluDerivative(const Matrix& m, double alpha = 0.01);

    // 辅助函数
    void randomize(double min = -0.5, double max = 0.5);
    void zeros();
    void ones();
    void printDimensions(const std::string& name) const {
        std::cout << name << " dimensions: " << rows << "x" << cols << std::endl;
    }
};

#endif //AI3700_MATRIX_H