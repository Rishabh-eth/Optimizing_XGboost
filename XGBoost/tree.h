//
// Created by Fizza Zafar on 09.04.20.
//

#ifndef XGBOOST_TREE_H
#define XGBOOST_TREE_H

#include "config.h"

typedef struct Node {
    double* instance_X;
    int* instance_pointers;
    double* instance_gh;
    long count_instances;

    double prediction_val;
    int depth;
    double split_val;
    int split_feature;

    struct Node* left;
    struct Node* right;
} Node;

typedef struct Queue {
    Node** qu;
    int head;
    int tail;
    int size;
    int capacity;
} Queue;

void pushQueue(Queue* qu, Node* r);

Node* popQueue(Queue* q); //remove Node from head of queue

typedef struct Tree {
    Node* root;
    int num_features;
    Queue* potential_split_nodes; //array of split nodes
    int depth;
    Config* config;
} Tree;

Node* createNode(double pred, double* X, int * pointers, double* gh, long nrows, int depth);

Tree* createTree(double* X, int* pointers, double* gh, int num_features, long nrows, Config* config);

Node* get_potential_split_node(Tree* tree);

void getInstancesLabels(double *X, int *pointers, long total_count, long total_left, double **left_instances,
                        int **left_pointers, long total_right, double **right_instances, int **right_pointers,
                        int num_features, const unsigned char *in_left_split, const int *new_pointer_mapping,
                        const double *gh, double** left_gh,double** right_gh,double *left_value,double *right_value);

void decide_branch(const double *X, const int *pointers, long total_count, int feature, double val, long *left_count,
                   long *right_count,unsigned char** in_left_split);


int *populate_metadata(long total_count,
                       long total_left, long total_right,
                       const unsigned char *in_left_split);

void split(Tree* tree, Node* node, int feature, double val);

void predict(Tree* tree, double* X, int n_rows, double* predictions);

void print(Node* node);

#endif //XGBOOST_TREE_H