//
// Created by Amory Hoste on 04/05/2020.
//

#ifndef XGBOOST_MATRIX_HELPERS_H
#define XGBOOST_MATRIX_HELPERS_H

void fill_array(double v, double n_rows, double* p);
void get_sorted_indices(double* X, int n_features, int n_rows, int* sorted_indices);
void get_sorted_matrix(const double* X, int n_features, int n_rows, const int* sorted_indices, double* sorted);
void generate_gh_matrix(const double* gh, int n_features, int n_rows, const int* sorted_indices, double* gh_mat);

#endif //XGBOOST_MATRIX_HELPERS_H
