//
// Created by Amory Hoste on 16/04/2020.
//

#ifndef WIN32
#include <sys/time.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "../config.h"
#include "../split.h"
#include "../matrix_helpers.h"
#include "tsc_x86.h"
#include "../tree.h"
#include "../classifier.h"

#define NUM_RUNS 1
#define CYCLES_REQUIRED 1e8
#define CALIBRATE
#define MAX_MAT_NUMBER 10000 // Largest number to appear in the matrix

// TODO: important if change way memory is allocated and passed somewhere also change in benchmark to avoid leaks / inaccurate measurements!

void fill_matrix_rand(double *mat, int rows, int cols) {
    for(int j=0; j < cols; j++) {
        for(int i=0; i < rows; i++) {
            mat[rows*j+i] = (double)rand()/(double)((double)RAND_MAX/MAX_MAT_NUMBER);
        }
    }
}

/**
 * Timing function based on the TimeStep Counter of the CPU.
 * Source: ASL homework 1
 */
double rdtsc_splitfinding(double mat[], double gh[], int rows, int cols, int ptrs[]) {
    int i, num_runs;
    myInt64 cycles;
    myInt64 start;
    num_runs = NUM_RUNS;

    /*
     * The CPUID instruction serializes the pipeline.
     * Using it, we can create execution barriers around the code we want to time.
     * The calibrate section is used to make the computation large enough so as to
     * avoid measurements bias due to the timing overhead.
     */
#ifdef CALIBRATE
    while(num_runs < (1 << 14)) {
        start = start_tsc();
        for (i = 0; i < num_runs; ++i) {
            split_finding(mat, ptrs, gh, cols, rows, LAMBDA);
        }
        cycles = stop_tsc(start);

        if(cycles >= CYCLES_REQUIRED) break;

        num_runs *= 2;
    }
#endif

    start = start_tsc();
    for (i = 0; i < num_runs; ++i) {
        split_finding(mat, ptrs, gh, cols, rows, LAMBDA);
    }
    cycles = stop_tsc(start) /num_runs;

    return (double) cycles;
}


double benchmark_splitfinding(int rows, int cols) {
    double* X = (double *)malloc(rows*cols*sizeof(double));
    double* gh = (double *)malloc(rows*2*sizeof(double));
    fill_matrix_rand(X, rows, cols); //matrix in column major order (Doesn't really matter but for consistency sake)
    fill_matrix_rand(gh,rows,2);

    int *pointers = (int *) malloc(rows * cols * sizeof(int));
    get_sorted_indices(X, cols, rows, pointers);

    double* column_sorted_mat = (double *) malloc(rows * cols * sizeof(double));
    double* col_sorted_gh_mat = (double *)malloc(2*rows*cols* sizeof(double));
    get_sorted_matrix(X, cols, rows, pointers, column_sorted_mat);
    generate_gh_matrix(gh, cols, rows, pointers, col_sorted_gh_mat);

    double cycles = rdtsc_splitfinding(column_sorted_mat, col_sorted_gh_mat, rows, cols, pointers);

    free(X);
    free(gh);
    free(column_sorted_mat);
    free(col_sorted_gh_mat);
    free(pointers);

    return cycles;
}

myInt64 get_split_cycles(double* X, int* pointers, double* gh, int num_features, long nrows, Config* config, int feature, double value) {
    myInt64 start;
    myInt64 cycles;

    // Prepare tree and node
    Tree* tree = createTree(X, pointers, gh, num_features, nrows, config);
    Node* split_node = get_potential_split_node(tree);

    // Time split
    start = start_tsc();
    split(tree, split_node, feature, value);
    cycles = stop_tsc(start);

    // Free tree and node, currently don't need to free elements in root node because not allocated in tree building
    free(split_node);
    split_node = get_potential_split_node(tree);
    while (split_node != NULL) {
        free(split_node->instance_X);
        free(split_node->instance_gh);
        free(split_node->instance_pointers);
        free(split_node);
        split_node = get_potential_split_node(tree);
    }
    free(tree->potential_split_nodes->qu);
    free(tree->potential_split_nodes);
    free(tree);

    return cycles;
}

/**
 * Timing function based on the TimeStep Counter of the CPU.
 * Source: ASL homework 1
 */
double rdtsc_split(double* X, int* pointers, double* gh, int num_features, long nrows, Config* config, int feature, double value) {
    int i, num_runs;
    myInt64 total_cycles;
    myInt64 cycles;
    num_runs = NUM_RUNS;

    /*
     * The CPUID instruction serializes the pipeline.
     * Using it, we can create execution barriers around the code we want to time.
     * The calibrate section is used to make the computation large enough so as to
     * avoid measurements bias due to the timing overhead.
     */
#ifdef CALIBRATE
    while(num_runs < (1 << 14)) {

        cycles = 0;
        for (i = 0; i < num_runs; ++i) {
            cycles += get_split_cycles(X, pointers, gh, num_features, nrows, config, feature, value);
        }

        if(cycles >= CYCLES_REQUIRED) break;

        num_runs *= 2;
    }
#endif

    total_cycles = 0;
    for (i = 0; i < num_runs; ++i) {
        total_cycles += get_split_cycles(X, pointers, gh, num_features, nrows, config, feature, value);
    }

    cycles = total_cycles /num_runs;

    return (double) cycles;
}

double benchmark_split(int rows, int cols) {
    double* X = (double *)malloc(rows*cols*sizeof(double));
    double* gh = (double *)malloc(rows*2*sizeof(double));
    fill_matrix_rand(X, rows, cols); //matrix in column major order (Doesn't really matter but for consistency sake)
    fill_matrix_rand(gh,rows,2);

    int *pointers = (int *) malloc(rows * cols * sizeof(int));
    get_sorted_indices(X, cols, rows, pointers);

    double* column_sorted_mat = (double *) malloc(rows * cols * sizeof(double));
    double* col_sorted_gh_mat = (double *)malloc(2*rows*cols* sizeof(double));
    get_sorted_matrix(X, cols, rows, pointers, column_sorted_mat);
    generate_gh_matrix(gh, cols, rows, pointers, col_sorted_gh_mat);

    // TODO: could also pass as command line params
    Config* config = (Config*) malloc(sizeof(Config));
    config->num_trees = NUM_TREES;
    config->max_depth = MAX_DEPTH;
    config->lambda = LAMBDA;
    config->learning_rate = LEARNING_RATE;
    config->gamma = GAMMA;
    config->min_instances = MIN_INSTANCES;

    // Take random col
    srand(time(NULL));
    int col = rand() % cols;
    double avg_value = 0;
    for (int r = 0; r < rows; r++) {
        avg_value += column_sorted_mat[col * rows + r] / rows;
    }
    double cycles = rdtsc_split(column_sorted_mat, pointers, col_sorted_gh_mat, cols, rows, config, col, avg_value);

    free(X);
    free(gh);
    free(column_sorted_mat);
    free(col_sorted_gh_mat);
    free(pointers);
    free(config);

    return cycles;
}

void free_tree(Node* cur, int clear_cur) {
    if (cur->left != NULL) {
        free_tree(cur->left, 1);
        free_tree(cur->right, 1);
    }

    if (clear_cur) {
        free(cur->instance_X);
        free(cur->instance_gh);
        free(cur->instance_pointers);
    }

    free(cur);
}

myInt64 get_tree_cycles(Config * config, Input* input) {
    myInt64 start;
    myInt64 cycles;

    // Prepare needed inputs
    Classifier* classifier = (Classifier*)malloc(sizeof(Classifier));
    classifier->config = config;

    // Time tree creation
    start = start_tsc();
    fit_classifier(classifier, input);
    cycles = stop_tsc(start);

    // Free tree
    for (int i = 0; i < classifier->num_trees; i++) {
        Tree* tree = classifier->trees[i];
        if (i == 0) { // Only clear top level X etc once since reused
            free_tree(classifier->trees[i]->root, 1);
        } else {
            free_tree(classifier->trees[i]->root, 0);
        }
        free(tree->potential_split_nodes->qu);
        free(tree->potential_split_nodes);
        free(tree);
    }

    free(classifier->trees);
    free(classifier);

    return cycles;
}


/**
 * Timing function based on the TimeStep Counter of the CPU.
 * Source: ASL homework 1
 */
double rdtsc_tree(Config * config, Input* input) {
    int i, num_runs;
    myInt64 total_cycles;
    myInt64 cycles;
    num_runs = NUM_RUNS;

    /*
     * The CPUID instruction serializes the pipeline.
     * Using it, we can create execution barriers around the code we want to time.
     * The calibrate section is used to make the computation large enough so as to
     * avoid measurements bias due to the timing overhead.
     */
#ifdef CALIBRATE
    while(num_runs < (1 << 14)) {

        cycles = 0;
        for (i = 0; i < num_runs; ++i) {
            cycles += get_tree_cycles(config, input);
        }

        if(cycles >= CYCLES_REQUIRED) break;

        num_runs *= 2;
    }
#endif

    total_cycles = 0;
    for (i = 0; i < num_runs; ++i) {
        total_cycles += get_tree_cycles(config, input);
    }

    cycles = total_cycles / num_runs;

    return (double) cycles;
}

double benchmark_tree(int rows, int cols) {
    // TODO: could also pass as command line params
    Config* config = (Config*) malloc(sizeof(Config));
    config->num_trees = NUM_TREES;
    config->max_depth = MAX_DEPTH;
    config->lambda = LAMBDA;
    config->learning_rate = LEARNING_RATE;
    config->gamma = GAMMA;
    config->min_instances = MIN_INSTANCES;

    double* X = (double *)malloc(rows*cols*sizeof(double));
    double* Y = (double *)malloc(rows*1*sizeof(double));
    fill_matrix_rand(X, rows, cols);
    fill_matrix_rand(Y, rows, 1);

    Input* input = (Input*)malloc(sizeof(Input));
    input->X = X;
    input->Y = Y;
    input->n_rows = rows;
    input->n_features = cols;

    double cycles = rdtsc_tree(config, input);

    free(config);
    free(X);
    free(Y);
    free(input);

    return cycles;
}

void run_benchmark(char* mode, int col_from, int col_to, int col_step, int row_from, int row_to, int row_step) {
    for (int c = col_from; c <= col_to; c+=col_step) {
        for (int r = row_from; r <= row_to; r+=row_step) {
            if (strcmp(mode, "all") == 0) {
                double c1 = benchmark_splitfinding(r, c);
                double c2 = benchmark_split(r, c);
                double c3 = benchmark_tree(r, c);
                printf("{'rows': %d, 'columns': %d, 'cycles_splitfinding': %lf, 'cycles_split': %lf, 'cycles_tree': %lf}\n",
                        r, c, c1, c2, c3);
            }
            else if (strcmp(mode, "splitfind") == 0) {
                double cycles = benchmark_splitfinding(r, c);
                printf("{'rows': %d, 'columns': %d, 'cycles': %lf}\n", r, c, cycles);
            } else if (strcmp(mode, "split") == 0) {
                double cycles = benchmark_split(r, c);
                printf("{'rows': %d, 'columns': %d, 'cycles': %lf}\n", r, c, cycles);
            } else if (strcmp(mode, "tree") == 0) {
                double cycles = benchmark_tree(r, c);
                printf("{'rows': %d, 'columns': %d, 'cycles': %lf}\n", r, c, cycles);
            }
        }
    }
}

int params_valid(int argc, char** argv) {
    if (argc == 8) {
        if (atoi(argv[6]) <= 0 || atoi(argv[7]) <= 0) {
            printf("row and col stepsizes must be >= 1\n");
            return 0;
        } else if (strcmp(argv[1], "splitfind") != 0 && strcmp(argv[1], "split") != 0
                    && strcmp(argv[1], "tree") != 0 && strcmp(argv[1], "all") != 0 ) {
            printf("mode '%s' is not a valid mode\n", argv[1]);
            return 0;
        } else {
            return 1;
        }
    } else {
        printf("Number of arguments do not match: %d\n", argc);
        return 0;
    }
}

int main(int argc, char** argv) {
    if (params_valid(argc, argv)) {
        run_benchmark(argv[1], atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7]));
        return 0;
    } else {
        printf("usage: benchmark mode col_from col_to col_step row_from row_to row_step\n");
        printf("> executes for all combinations of [col_from ... col_to] and [row_from ... row_to] with given row and col stepsizes\n");
        printf("> valid modes:\n  * splitfind\n  * split\n  * tree\n  * all\n");
        return 1;
    }
}
