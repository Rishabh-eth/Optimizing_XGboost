//
// Created by Amory Hoste on 15/04/2020.
//

#ifndef XGBOOST_CONFIG_H
#define XGBOOST_CONFIG_H

// Default values
#define NUM_TREES 1
#define MAX_DEPTH 4 // XGBoost default 6 (in our implementation 7, maybe we should adjust our depths)

#define LAMBDA 10 // XGBoost default 1
#define LEARNING_RATE 1 // XGBoost default: 0.3 (eta)
#define GAMMA 0 // XGBoost default: 0

#define MIN_INSTANCES 8

typedef struct Config {
    int num_trees;
    int max_depth;
    int lambda;
    double learning_rate;
    int gamma;
    int min_instances;
} Config;

#endif //XGBOOST_CONFIG_H
