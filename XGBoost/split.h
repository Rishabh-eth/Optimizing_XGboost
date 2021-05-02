//
// Created by Noman Sheikh on 13/04/2020.
//

#ifndef XGBOOST_SPLIT_H
#define XGBOOST_SPLIT_H

struct splitFindingResult {
    double gain;
    double value;
    int feature;
};

struct splitFindingResult split_finding(const double* X, const int* pointers, const double* gh, int num_features, int num_rows, double lambda);

#endif //XGBOOST_SPLIT_H
