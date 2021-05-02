//
// Created by Amory Hoste on 15/04/2020.
//

#include "libs/catch.hpp"
#include "libs/json.hpp"
#include "transformers.h"
#include <vector>
#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <stdexcept>
#include <iostream>

#define EPSILON 0.0001

extern "C" {
    #include "../inputreader.h"
    #include "../classifier.h"
    #include "../tree.h"
    #include "../config.h"
}

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    std::unique_ptr<char[]> buf( new char[ size ] );
    snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

bool doubles_similar (double d1, double d2) {
    return fabs(d1 - d2) < EPSILON;
}

using json = nlohmann::json;

bool contains_key(json j_tree, std::string key) {
    return j_tree.find(key) != j_tree.end();
}

bool correct_tree(json j_tree, Node* node) {
    bool tree_correct = true;
    bool node_correct = true;
    std::string error = "";

    int depth = j_tree["depth"];

    if (node->depth != depth) {
        error.append(string_format(" - Depth should be %d but it is %d\n", depth, node->depth));
        node_correct = false;
    }

    if (node->left == NULL) {
        error.append(string_format("Node - Instances: %ld, Depth: %d, Log-odds: %f \n",
                node->count_instances, node->depth, node->prediction_val));

        if (!contains_key(j_tree, "prediction_val")) {
            error.append(" - should not be a leaf node but it is\n");
            node_correct = false;
        } else {
            double prediction_val = j_tree["prediction_val"];

            if (! doubles_similar(node->prediction_val, prediction_val)) {
                error.append(string_format(" - Log-odds should be %f but it is %f\n", prediction_val, node->prediction_val));
                node_correct = false;
            }
        }


    } else {
        error.append(string_format("Node - Feature: %d, Val: %f, Instances: %ld, Depth: %d, Log-odds: %f \n",
                node->split_feature, node->split_val, node->count_instances, node->depth, node->prediction_val));

        if (!contains_key(j_tree, "split_feature") || !contains_key(j_tree, "split_val") || !contains_key(j_tree, "left") || !contains_key(j_tree, "right")) {
            error.append(" - should be be a leaf node but it is not\n");
            node_correct = false;
        } else {
            int split_feature = j_tree["split_feature"];
            double split_val = j_tree["split_val"];
            json left = j_tree["left"];
            json right = j_tree["right"];

            if (node->split_feature != split_feature) {
                error.append(string_format(" - Feature should be %d but it is %d\n", split_feature, node->split_feature));
                node_correct = false;
            }

            if (! doubles_similar(node->split_val, split_val)) {
                error.append(string_format(" - Feature should be %f but it is %f\n", split_val, node->split_val));
                node_correct = false;
            }

            tree_correct = tree_correct and correct_tree(left, node->left);
            tree_correct = tree_correct and correct_tree(right, node->right);
        }
    }

    if (!node_correct) {
        std::cerr << error << std::endl;
    }

    return tree_correct and node_correct;
}

void test_tree(std::string path) {
    // Read JSON input
    std::ifstream in(path);
    json j;
    in >> j;

    for (auto el : j) {

        // Parse json
        std::vector<double> orig_x = el["x"];
        int n_rows = el["rows"];
        int n_cols = el["cols"];

        std::vector<double> x(n_rows * n_cols);
        convert_column_major(orig_x, n_rows, n_cols, x);

        std::vector<double> Y = el["Y"];

        // Create input
        Input* input = (Input*)malloc(sizeof(Input));

        double* x_arr = (double *) malloc(n_rows*n_cols*sizeof(double));
        std::copy(x.begin(), x.end(), x_arr);
        input->X = x_arr;

        double* Y_arr = (double*)malloc(n_rows*sizeof(double));
        std::copy(Y.begin(), Y.end(), Y_arr);
        input->Y = Y_arr;

        input->n_features = n_cols;
        input->n_rows = n_rows;

        int n_trees = el["n_trees"];
        int max_depth = el["max_depth"];
        int lambda = el["lambda"];
        double learning_rate = el["learning_rate"];
        int gamma = el["gamma"];
        int min_instances = el["min_instances"];

        Config* config = (Config*) malloc(sizeof(Config));
        config->num_trees = n_trees;
        config->max_depth = max_depth;
        config->lambda = lambda;
        config->learning_rate = learning_rate;
        config->gamma = gamma;
        config->min_instances = min_instances;

        Classifier* classifier = (Classifier*)malloc(sizeof(Classifier));
        classifier->config = config;

        // Execute function
        fit_classifier(classifier, input);

        int i = 0;
        for (json j_tree : el["trees"]) {
            // CAPTURE(n_rows, n_cols, x, Y, ini_pred);
            CAPTURE(n_rows, n_cols);
            bool correct = correct_tree(j_tree, classifier->trees[i]->root);
            REQUIRE(correct);
            i++;
        }

        // Free
        free(x_arr);
        free(Y_arr);
        free(input);
        free(config);
        free(classifier);
    }
}

TEST_CASE("Correctly creates one tree using exact greedy with 10 thousand rows") {
    test_tree("../test/data/one_tree_exact_greedy_10k.json");
}

TEST_CASE("Correctly creates multiple trees using exact greedy with 10 thousand rows") {
    test_tree("../test/data/multiple_tree_exact_greedy_10k.json");
}

TEST_CASE("Correctly creates multiple trees using exact greedy with 50 thousand rows") {
    test_tree("../test/data/multiple_tree_exact_greedy_50k.json");
}

TEST_CASE("Correctly creates multiple trees using exact greedy with 100 thousand rows") {
    test_tree("../test/data/multiple_tree_exact_greedy_100k.json");
}
