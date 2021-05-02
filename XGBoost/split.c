//
// Created by Noman Sheikh on 12/04/20.
//


#include "split.h"

struct splitFindingResult split_finding(const double* X, const int* pointers, const double* gh, int num_features, int num_rows, double lambda) {

    int GRAD_COL = 0;
    int HESS_COL = 1;

    double G=0,H=0,G1=0,H1=0,G2=0,H2=0,G3=0,H3=0;
    int i;
    for (i=0; i<(num_rows-4); i+=4) {
        G += gh[num_rows * GRAD_COL + i];
        G1 += gh[num_rows * GRAD_COL + i+1];
        G2 += gh[num_rows * GRAD_COL + i+2];
        G3 += gh[num_rows * GRAD_COL + i+3];
        H += gh[num_rows * HESS_COL + i];
        H1 += gh[num_rows * HESS_COL + i+1];
        H2 += gh[num_rows * HESS_COL + i+2];
        H3 += gh[num_rows * HESS_COL + i+3];
    }
    for (;i<num_rows; i++) {
        G += gh[num_rows * GRAD_COL + i];
        H += gh[num_rows * HESS_COL + i];
    }

    G+=G1;
    G2+=G3;
    H+=H1;
    H2+=H3;
    G+=G2;
    H+=H2;

    double init_score = ((G*G) / (H + lambda));
    double best_score = 0;
    double best_split_val = -1;
    int best_split_feature = -1;
    for(int k=0; k<num_features; k++) {
        double GL=0, HL=0, GR=0, HR=0;
        double prev_score = 0;
        double prev_val = -1;
        for(i=0; i<num_rows; i++) {
            /*GL += gh[num_rows * GRAD_COL + pointers[k * num_rows + i]];
            HL += gh[num_rows * HESS_COL + pointers[k * num_rows + i]];*/

            GL+=gh[2*k*num_rows + i];
            HL+=gh[(2*k+1)*num_rows+i];

            GR = G - GL;
            HR = H - HL;
            double this_score = ((GL*GL) / (HL + lambda)) + ((GR*GR) / (HR + lambda)) - init_score;
            double Xk = X[k * num_rows + i];
            if (prev_score > best_score && Xk != prev_val) {
                best_score = prev_score;
                best_split_val = prev_val;
                best_split_feature = k;
            }
            prev_score = this_score;
            prev_val = Xk;
        }
    }
    struct splitFindingResult result;
    result.gain = best_score;
    result.value = best_split_val;
    result.feature = best_split_feature;
    return result;
}

