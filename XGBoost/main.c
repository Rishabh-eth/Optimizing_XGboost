#include <stdlib.h>
#include <stdio.h>
#include "tree.h"
#include "inputreader.h"
#include "classifier.h"
#include "config.h"

#define FEATURE_PARAM_OFFSET 3

void classify(Input* input) {
    Config* config = (Config*) malloc(sizeof(Config));
    config->num_trees = NUM_TREES;
    config->max_depth = MAX_DEPTH;
    config->lambda = LAMBDA;
    config->learning_rate = LEARNING_RATE;
    config->gamma = GAMMA;
    config->min_instances = MIN_INSTANCES;

    Classifier* classifier = (Classifier*)malloc(sizeof(Classifier));
    classifier->config = config;

    fit_classifier(classifier, input);
    print(classifier->trees[0]->root);

    free(classifier);
    free(config);
}

int main(int argc, char *argv[]) {

    if (argc >= 4) {
        // Parse parameters
        char* path = argv[1];

        // Read column that contains the labels
        int label_col = (int) strtol(argv[2], (char **)NULL, 10);

        // Read all columns containing the features into an array
        int n_features = argc - FEATURE_PARAM_OFFSET;
        int* feature_cols = (int *)malloc(n_features * sizeof(int));
        for (int i = 0; i < n_features; i++) {
            feature_cols[i] = (int) strtol(argv[i + FEATURE_PARAM_OFFSET], (char **)NULL, 10);
        }

        // Read input
        Input* input = (Input*)malloc(sizeof(Input));

        if (read_csv(path, label_col, n_features, feature_cols, input) != 0) {
            printf("\n file opening failed ");
            return -1 ;
        }

        classify(input);

        // Free data
        free(input->X);
        free(input->Y);
        free(input);
        free(feature_cols);

        return 0;
    } else {
        printf("usage: XGBoost 'csv-path' label-column feature-column1 feature-column2 ...'\n");
        return 1;
    }
}
