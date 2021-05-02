//
// Created by Amory Hoste on 27/04/2020.
//

#include "transformers.h"

void convert_column_major(std::vector<double> &in, int n_rows, int n_cols, std::vector<double> &out) {
    for (int r = 0; r < n_rows; r++) {
        for (int c = 0; c < n_cols; c++) {
            out[c * n_rows + r] = in[r * n_cols + c];
        }
    }
}