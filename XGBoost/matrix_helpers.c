//
// Created by Amory Hoste on 04/05/2020.
//

#include "matrix_helpers.h"
#include "sort.h"
#include <stdlib.h>
#include "string.h"

/**
 * Fills an array with a certain value
 * @param base_pred the initial prediction
 * @param n_rows the amount of rows in the array
 * @param p array to write the value to
 */
void fill_array(double v, double n_rows, double* p) {
    for (int i = 0; i < n_rows; i++) {
        p[i] = v;
    }
}

/**
 * Calculates a matrix containing the sorted indices of all columns of matrix X
 * @param X the matrix to calculate the sorted indices for
 * @param n_features the amount of features in X
 * @param n_rows the amount of rows in X
 * @param sorted_indices the matrix to write the sorted indices to
 */
void get_sorted_indices(double* X, int n_features, int n_rows, int* sorted_indices) {
    int* sorted = (int*) malloc(n_rows * sizeof(int));
    for (int i = 0; i < n_features; i++) {
        sorted_data(X + i * n_rows, n_rows, sorted);
        memcpy(sorted_indices + i * n_rows, sorted, n_rows * sizeof(int));
    }
    free(sorted);
}

/**
 * Sorts a matrix X column by column based on a matrix of indices that indicates how the columns should be sorted
 * @param X the matrix to sort
 * @param n_features the amount of features in X
 * @param n_rows the amount of rows in X
 * @param sorted_indices a matrix that indicates how to sort the columns
 * @param sorted a columnwise sorted matrix
 */
void get_sorted_matrix(const double* X, int n_features, int n_rows, const int* sorted_indices, double* sorted) {
    for (int c = 0; c < n_features; c++) {
        for (int r = 0; r < n_rows; r++) {
            sorted[c * n_rows + r] = X[c * n_rows + sorted_indices[c * n_rows + r]];
        }
    }
}

void generate_gh_matrix(const double* gh, int n_features, int n_rows, const int* sorted_indices, double* gh_mat) {
    for (int c = 0; c < n_features; c++) {
        for (int r = 0; r < n_rows; r++) {
            gh_mat[2*c*n_rows + r] = gh[sorted_indices[c * n_rows + r]];
            gh_mat[(2*c+1)*n_rows + r] = gh[n_rows + sorted_indices[c * n_rows + r]];
        }
    }
}