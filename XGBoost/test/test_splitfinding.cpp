//
// Created by Amory Hoste on 13/04/2020.
//

#include "libs/catch.hpp"
#include "libs/json.hpp"
#include "transformers.h"
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>

extern "C" {
    #include "../split.h"
    #include "../sort.h"
}

/**
 * Adds columns to an existing matrix in vector form
 * @param matrix a matrix in vector form to add the columns to
 * @param n_rows the amount of rows of the original matrix
 * @param n_cols the amount of columns of the original matrix
 * @param columns the columns to add to the original matrix
 * @param result_matrix the matrix to write the results to
 */
void add_columns(std::vector<double> &matrix, int n_rows, int n_cols, std::vector<std::vector<double>> &columns, std::vector<double> &result_matrix) {
    result_matrix.reserve(n_rows * (n_cols + columns.size()));

    for (int row = 0; row < n_rows; row++) {
        std::vector<double> row_vec; row_vec.reserve(n_cols);

        // Add row to 2D vector
        row_vec.insert(row_vec.end(), matrix.begin() + (row * n_cols), matrix.begin() + (row * n_cols) + n_cols);

        // Add columns to 2D vector
        for (std::vector<double> column : columns) {
            row_vec.push_back(column[row]);
        }

        result_matrix.insert(result_matrix.end(), row_vec.begin(), row_vec.end());
    }
}

/************************************************
 *                     TESTS                    *
 ************************************************/
TEST_CASE("Correctly splits a predefined matrix") {
    using Catch::Matchers::Equals;

    // Initialize Inputs
    int n_rows = 5;
    int n_cols = 1;
    int feature = 0;
    int lambda = 10;

    SECTION("matrix 1") {

        std::vector<double> input_matrix = {
                1, 2, 3, 4, 5
        };

        std::vector<double> gh = {
                0.1, 0.2, 0.3, 0.4, 0.5,
                0.01, 0.02, 0.03, 0.04, 0.05
        };

        std::vector<int> sorted_indices = {
                1, 3, 0, 2, 5
        };

        // Execute function
        struct splitFindingResult result = split_finding(input_matrix.data(), sorted_indices.data(), gh.data(), n_cols, n_rows, lambda);

        // Check results
        CAPTURE(n_rows, n_cols, feature, lambda, input_matrix); // Log relevant variables if fails
        REQUIRE(result.value == -1); // TODO
        REQUIRE(result.feature == -1); // TODO
    }

    SECTION("matrix 2") {
        std::vector<double> input_matrix = {
                1, 2, 3, 4, 5
        };

        std::vector<double> gh = {
                0.2,0.4,0.1,-0.3,-0.4,
                1, 1, 1, 1, 1
        };

        std::vector<int> sorted_indices = {
                1, 3, 0, 2, 5
        };

        // Execute function
        struct splitFindingResult result = split_finding(input_matrix.data(), sorted_indices.data(),gh.data(), n_cols, n_rows, lambda);

        // Check results
        CAPTURE(n_rows, n_cols, feature, lambda, input_matrix, result.gain); // Log relevant variables if fails
        REQUIRE(result.value == 3); // TODO
        REQUIRE(result.feature == 0); // TODO
    }
}

void presort(double* mat, double* gh, int n_rows, int n_features, int* pointers, double * column_sorted_mat, double* column_sorted_gh) {

    for (int i = 0; i < n_features; i++) {
        int* sorted_indices = (int*) malloc(n_rows * sizeof(int));
        sorted_data(mat + i * n_rows, n_rows, sorted_indices);
        memcpy(pointers + i * n_rows, sorted_indices, n_rows * sizeof(int));
        free(sorted_indices);
    }

    for (int c = 0; c < n_features; c++) {
        for (int r = 0; r < n_rows; r++) {
            column_sorted_mat[c * n_rows + r] = mat[c * n_rows + pointers[c * n_rows + r]];
            column_sorted_gh[2*c*n_rows + r] = gh[pointers[c * n_rows + r]];
            column_sorted_gh[(2*c+1)*n_rows + r] = gh[n_rows + pointers[c * n_rows + r]];
        }
    }
}

TEST_CASE("Correctly splits matrices from the exact greedy json file") {

    // Read JSON input
    using json = nlohmann::json;
    std::ifstream in("../test/data/split_exact_greedy.json");
    json j;
    in >> j;
    int nr = 0;

    for (auto el : j) {
        nr++;

        // Parse json
        std::vector<double> orig_x = el["x"];
        int n_rows = el["rows"];
        int n_features = el["cols"];
        int lambda = el["lambda"];

        std::vector<double> x(n_rows * n_features);
        convert_column_major(orig_x, n_rows, n_features, x);

        std::vector<double> orig_gh = el["gh"];
        std::vector<double> gh(n_rows * 2);
        convert_column_major(orig_gh, n_rows, 2, gh);

        int *pointers = (int *) malloc(n_rows * n_features * sizeof(int));
        double* column_sorted_mat = (double *) malloc(n_rows * n_features * sizeof(double));

        double* column_sorted_gh = (double *) malloc(2* n_rows * n_features * sizeof(double));
        presort(x.data(), gh.data(),  n_rows, n_features, pointers, column_sorted_mat, column_sorted_gh);

        // Parse expected values
        double gain = el["gain"];
        int feature = el["feature"];
        double value = el["value"];

        // Execute function
        struct splitFindingResult result = split_finding(column_sorted_mat, pointers, column_sorted_gh, n_features, n_rows, lambda);

        // Check results
        //CAPTURE(gh,gh[pointers[0]],gh[pointers[1]],gh[pointers[2]],gh[pointers[3]],gh[pointers[4]]);
        //CAPTURE(n_rows, n_features, lambda, x); // Log relevant variables if fails
        CAPTURE(n_rows, n_features, lambda);
        REQUIRE(result.gain == Approx(gain));
        REQUIRE(result.value == value);
        REQUIRE(result.feature == feature);

        free(pointers);
        free(column_sorted_mat);

    }
}
