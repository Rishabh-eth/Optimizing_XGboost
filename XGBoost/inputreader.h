//
// Created by Rishabh Singh on 14/04/2020.
//

#ifndef XGBOOST_INPUTREADER_H
#define XGBOOST_INPUTREADER_H

typedef struct Input {
    double *X;
    int n_rows;
    int n_features;
    double *Y;
} Input;

int read_csv(char *path, int label_col, int n_features, int *feature_cols, struct Input *result);

#endif //XGBOOST_INPUTREADER_H
