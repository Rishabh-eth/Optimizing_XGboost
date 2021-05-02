//
// Created by Amory Hoste on 16/04/2020.
//

#ifndef XGBOOST_CLASSIFIER_H
#define XGBOOST_CLASSIFIER_H

#include "inputreader.h"
#include "tree.h"
#include "config.h"

typedef struct Classifier {
    double base_pred;
    Tree** trees;
    int num_trees;
    Config* config;
} Classifier;

void fit_classifier(Classifier* classifier, Input *input);

double* classifier_predict(Classifier* classifier, Input* input);

#endif //XGBOOST_CLASSIFIER_H
