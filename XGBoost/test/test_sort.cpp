//
// Created by Amory Hoste on 12/04/2020.
//

#include "libs/catch.hpp"
#include <vector>
#include <algorithm>

#include "pseudorandom.h"

extern "C" {
    #include "../sort.h"
}

/**
 * Generates a pseudo random matrix (in vector form)
 * @param n_rows the amount of rows in the matrix
 * @param n_cols the amount of columns in the matrix
 * @return a vector containing a pseudorandom matrix with 'n_row' rows and 'n_col' columns
 */
std::vector<double> generate_input_matrix(int n_rows, int n_cols) {
    std::vector<double> input(n_rows * n_cols);
    std::generate(input.begin(), input.end(), []() {
        return pseudorandom::get_random_double(pseudorandom::get_generator());
    });
    return input;
}

/**
 * Extracts a column from a matrix in vector form
 * @param source the matrix to extract the column from
 * @param n_rows the amount of rows of the matrix
 * @param n_cols the amount of columns of the matrix
 * @param col the column to extract from the matrix
 * @return a vector containing the column with index 'col' in the source matrix
 */
std::vector<double> get_column(std::vector<double> &source, int n_rows, int col) {
    std::vector<double> result; result.reserve(n_rows);
    for (int i = 0; i < n_rows; i++) {
        result.push_back(source[col * n_rows + i]);
    }
    return result;
}

/**
 * Extracts all values at given indices from an input vector
 * @param source the vector to extract the values from
 * @param indices the indices of the values to extract
 * @param n the length of the indices vector
 * @return a vector with all values at the given indices
 */
std::vector<double> get_values(std::vector<double> &source, std::vector<int> &indices, int n) {
    std::vector<double> result; result.reserve(n);
    for (int i = 0; i < n; i++) {
        result.push_back(source[indices[i]]);
    }
    return result;
}

/**
 * Creates a vector of length n starting from 'start' with stepsize 'step'
 * @param start the number to start from
 * @param step the stepsize
 * @param n the amount of elements in the resulting vector
 * @return a vector of length n starting from 'start' with stepsize 'step'
 */
int* range(int start, int step, int n) {
    int* indices = (int*) malloc(n * sizeof(int));
    for(int i=0; i<n; i++) {
        indices[i] = start + i*step;
    }
    return indices;
}

/**
 * Runs mergesort on a column to create a vector of sorted indices
 * @param input_matrix the matrix to run mergesort on
 * @param n_rows the amount of rows of the matrix
 * @param n_cols the amount of columns of the matrix
 * @param feature the feature to sort on
 * @return a vector containing indices to sort the 'feature' column
 */
std::vector<int> run_mergesort_idx(std::vector<double> input_matrix, int n_rows, int feature) {
    int* indices_arr = range(0, 1, n_rows);
    mergeSort(indices_arr, input_matrix.data() + n_rows * feature, 0, n_rows-1);
    std::vector<int> result_indices(indices_arr, indices_arr+n_rows);
    free(indices_arr);

    return result_indices;
}

/**
 * Runs mergesort on a column using the c++ sorting algorithm to create a vector of sorted column values
 * @param input_matrix the matrix to sort
 * @param n_rows the amount of rows of the matrix
 * @param n_cols the amount of columns of the matrix
 * @param feature the feature to sort on
 * @return a vector of sorted column values
 */
std::vector<double> mergesort_expected(std::vector<double> input_matrix, int n_rows, int feature) {
    std::vector<double> expected_values(input_matrix.begin() +  feature * n_rows, input_matrix.begin() +  feature * n_rows + n_rows);
    std::sort(expected_values.begin(), expected_values.end());
    return expected_values;
}

/************************************************
 *                     TESTS                    *
 ************************************************/
TEST_CASE("Correctly creates sorted indices for a column of predefined matrix") {
    using Catch::Matchers::Equals;

    // Initialize Inputs
    int feature = 0;
    int n_features = 3;
    int n_rows = 5;

    std::vector<double> input_matrix = {
            3, 1, 4, 2, 5,
            0.1, 0.2, 0.3, 0.4, 0.5,
            0.01, 0.02, 0.03, 0.04, 0.05
    };

    // Execute function
    std::vector<int> result_indices = run_mergesort_idx(input_matrix, n_rows, feature);

    // Check results
    std::vector<int> expected_indices = {1, 3, 0, 2, 4};

    CAPTURE(n_rows, n_features, feature, expected_indices, result_indices); // Log relevant variables if fails
    REQUIRE_THAT(result_indices, Equals(expected_indices));
}

TEST_CASE("Correctly sorts columns of a random matrix") {
    using Catch::Matchers::Equals;

    // Initialize Inputs, test for all combinations of n_rows and n_cols below
    auto n_rows = GENERATE(1, 2, 5, 10, 50, 100, 500, 999);
    auto n_features = GENERATE(1, 2, 5, 10);
    std::vector<double> input_matrix = generate_input_matrix(n_rows, n_features);

    SECTION("All columns correctly sorted" ) {
        // Test sorting for each feature
        for (int feature = 0; feature < n_features; feature++) {
            // Execute function
            std::vector<int> result_indices = run_mergesort_idx(input_matrix, n_rows, feature);

            // Check results
            std::vector<double> column = get_column(input_matrix, n_rows, feature);
            std::vector<double> result_values = get_values(column, result_indices, n_rows);
            std::vector<double> expected_values = mergesort_expected(input_matrix, n_rows, feature);

            CAPTURE(n_rows, n_features, feature, result_values, expected_values);
            REQUIRE_THAT(result_values, Equals(expected_values));
        }
    }
}

TEST_CASE("Correctly sorts a column of large matrix") {
    using Catch::Matchers::Equals;

    // Initialize Inputs, test for all combinations of n_rows and n_cols below
    int n_rows = 2500000;
    int n_features = 3;
    std::vector<double> input_matrix = generate_input_matrix(n_rows, n_features);

    int feature = 1;

    // Execute function
    std::vector<int> result_indices = run_mergesort_idx(input_matrix, n_rows, feature);

    // Check results
    std::vector<double> column = get_column(input_matrix, n_rows, feature);
    std::vector<double> result_values = get_values(column, result_indices, n_rows);
    std::vector<double> expected_values = mergesort_expected(input_matrix, n_rows, feature);

    // CAPTURE(n_rows, n_features, feature, result_values, expected_values);
    CAPTURE(n_rows, n_features, feature);
    REQUIRE_THAT(result_values, Equals(expected_values));
}