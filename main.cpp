#include <iostream>
#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <ctime>

const int N = 300; // Размерность матрицы

double randomDouble() {
    return (2.0 * rand() / RAND_MAX) - 1.0;
}

double** fillRandomMatrix() {
    srand(time(0));


    double** mat = new double*[N];
    for (int i = 0; i < N; ++i) {
        mat[i] = new double[N + 1];
        for (int j = 0; j < N + 1; ++j) {
            mat[i][j] = randomDouble();
        }
    }

    return mat;
}

void deleteMatrix(double** mat) {
    for (int i = 0; i < N; ++i) {
        delete[] mat[i];
    }
    delete[] mat;
}

void setExtraVarsToNull(double** mat) {
    int max_row;
    double temp;

    for (int i = 0; i < N - 1; i++) {
        #pragma omp parallel for shared(mat) private(max_row, temp)
        for (int k = i + 1; k < N; k++) {
            max_row = i;
            for (int j = i + 1; j < N; j++) {
                if (fabs(mat[j][i]) > fabs(mat[max_row][i]))
                    max_row = j;
            }
            if (max_row != i) {
                for (int j = i; j <= N; j++) {
                    temp = mat[i][j];
                    mat[i][j] = mat[max_row][j];
                    mat[max_row][j] = temp;
                }
            }
            double pivot = mat[i][i];
            #pragma omp parallel for shared(mat) private(temp)
            for (int j = i + 1; j < N; j++) {
                double factor = mat[j][i] / pivot;
                for (int k = i; k <= N; k++) {
                    mat[j][k] -= factor * mat[i][k];
                }
            }
        }
    }
}

void getResult(double** mat, double result[N]) {
    for (int i = N - 1; i >= 0; i--) {
        result[i] = mat[i][N];
        for (int j = i + 1; j < N; j++) {
            result[i] -= mat[i][j] * result[j];
        }
        result[i] = result[i] / mat[i][i];
    }
}

void gaussianMethod(double** mat, double result[N]) {
    setExtraVarsToNull(mat);
    getResult(mat, result);
}

int main() {
    double** mat = fillRandomMatrix();
    double result[N];

    gaussianMethod(mat, result);

    std::cout << "Solution:\n";
    for (int i = 0; i < N; i++) {
        std::cout << "x" << i << " = " << result[i] << std::endl;
    }

    deleteMatrix(mat);

    return 0;
}

