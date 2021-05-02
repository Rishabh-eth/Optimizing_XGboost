//
// Created by Amory Hoste on 16/04/2020.
//
#include "classifier.h"
#include "matrix_helpers.h"
#include "split.h"
#include <stdlib.h>
#include "string.h"
#include <math.h>

/**
 * Calculates the gradient and hessian using logloss as loss function
 * @param p array containing the predicted probabilities
 * @param Y array containing the labels
 * @param n_rows the amount of rows in the matrix
 * @param gh array to write the results to
 */
void calculate_grad_hess_logloss(const double* p, const double* Y, int n_rows, double* gh) {
    for(int i = 0; i < n_rows; i++) {
        gh[i] = p[i]- Y[i];
        gh[n_rows + i] = p[i]*(1-p[i]);
    }
}

/**
 * Calculates the initial prediction
 * @param Y array containing the labels
 * @param n_rows the amount of rows in the matrix
 * @return the initial prediction
 */
double get_base_predediction(const double* Y, int n_rows) {
    double sum = 0;

    for(int i=0; i < n_rows; i++) {
        sum = sum + Y[i];
    }

    return sum / n_rows;
}

void fit_classifier(Classifier* classifier, Input* input) {

    // Calculate base prediction
    double base_pred = get_base_predediction(input->Y, input->n_rows);
    classifier->base_pred = base_pred;

    // Presorting step
    int *pointers = (int *) malloc(input->n_rows * input->n_features * sizeof(int));
    get_sorted_indices(input->X, input->n_features, input->n_rows, pointers);

    double* column_sorted_mat = (double *) malloc(input->n_rows * input->n_features * sizeof(double));
    get_sorted_matrix(input->X, input->n_features, input->n_rows, pointers, column_sorted_mat);

    // Calculate initial probabilities
    double* p = (double*)malloc(input->n_rows * sizeof(double));
    fill_array(base_pred, input->n_rows, p);

    // Build trees
    classifier->num_trees = classifier->config->num_trees;
    classifier->trees = (Tree**) malloc(classifier->config->num_trees * sizeof(Tree*));
    double* gh = (double*)malloc(2*input->n_rows * sizeof(double));
    double* col_sorted_gh_mat = (double *)malloc(2*input->n_rows*input->n_features* sizeof(double));
    double* predictions = (double *) malloc(input->n_rows * sizeof(double));

    for (int t = 0; t < classifier->config->num_trees; t++) {
        // Calculate gradients and hessians
        calculate_grad_hess_logloss(p, input->Y, input->n_rows, gh);
        generate_gh_matrix(gh, input->n_features, input->n_rows, pointers, col_sorted_gh_mat);

        // Build tree
        Tree* tree = createTree(column_sorted_mat, pointers, col_sorted_gh_mat, input->n_features, input->n_rows, classifier->config);
        Node* split_node = get_potential_split_node(tree);
        while (split_node != NULL) {
            if (split_node->depth < classifier->config->max_depth) {
                struct splitFindingResult best_split = split_finding(split_node->instance_X, split_node->instance_pointers, split_node->instance_gh ,tree->num_features, (int) split_node->count_instances, classifier->config->lambda);

                if (best_split.feature != -1 && best_split.gain >= classifier->config->gamma) {
                    split(tree, split_node, best_split.feature, best_split.value);
                }
            }
            split_node = get_potential_split_node(tree);
        }

        // Update predictions
        predict(tree, input->X, input->n_rows, predictions);
        for (int i = 0; i < input->n_rows; i++) {
            p[i] += classifier->config->learning_rate * (1 / (1 + exp(-predictions[i])));
        }
        classifier->trees[t] = tree;
    }
    free(gh);
    free(p);
    free(predictions);
}

double* classifier_predict(Classifier* classifier, Input* input) {

    // Calculate initial prediction
    double* pred = (double*)malloc(input->n_rows * sizeof(double));
    double base_pred_log_odds = (1 / (1 + exp(-classifier->base_pred)));
    fill_array(base_pred_log_odds, input->n_rows, pred);

    double * predictions = (double *) malloc(input->n_rows * sizeof(double));
    for (int t = 0; t < classifier->num_trees; t++) {
        predict(classifier->trees[t], input->X, input->n_rows, predictions);
        for (int i = 0; i < input->n_rows; i++) {
            pred[i] += classifier->config->learning_rate * predictions[i];
        }
    }

    free(predictions);

    // Convert prediction from log odds to probability
    for (int i = 0; i < input->n_rows; i++) {
        pred[i] = (1 / (1 + exp(-pred[i])));
    }

    return pred;
}