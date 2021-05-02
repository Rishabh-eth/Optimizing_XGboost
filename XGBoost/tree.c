//
// Created by Fizza Zafar on 09.04.20.
//

#include "tree.h"

#include <stdio.h>
#include <stdlib.h>
#include "string.h"
#include "config.h"
#include <math.h>

void pushQueue(Queue* qu, Node* r) {
    if (qu->size == qu -> capacity) {
        int capacity = qu->capacity;
        qu->qu = realloc(qu->qu, 2*capacity*sizeof(Node*));
        qu -> capacity = 2*capacity;
    }
    qu->qu[qu->tail + 1] = r;
    (qu->size)++;
    qu->tail = (qu->tail + 1)%(qu->capacity);
}

Node* popQueue(Queue* q) { //remove Node from head of queue
    if (q->size > 0) {
        Node *head = q->qu[q->head];
        q->head = ((q->head) + 1) % (q->capacity);
        (q->size)--;
        return head;
    }
    return NULL;
}

Node* createNode(double pred, double* X, int * pointers, double* gh, long nrows, int depth) {
    Node* new_node = (Node*)malloc(sizeof(Node));
    new_node->instance_X = X;
    new_node->instance_gh = gh;
    new_node->instance_pointers = pointers;
    new_node->depth = depth;
    new_node->prediction_val = pred;
    new_node->left = NULL;
    new_node->right = NULL;
    new_node->count_instances = nrows;
    return new_node;
}

Tree* createTree(double* X, int* pointers, double* gh, int num_features, long nrows, Config* config) {
    Tree* new_tree = (Tree*)malloc(sizeof(Tree));
    new_tree->config = config;
    new_tree->num_features=num_features;

    double grads = 0, hess = 0;
    for (int i = 0; i < nrows; i++) {
        grads += gh[i];
        hess += gh[nrows+i];
    }
    double pred = - grads/(hess + config->lambda);

    new_tree->root = createNode(pred, X, pointers, gh, nrows, 1);
    new_tree->depth=1;
    new_tree->potential_split_nodes = (Queue*)malloc(sizeof(Queue));
    new_tree->potential_split_nodes->qu = (Node**)malloc(pow(2,config->max_depth) * sizeof(Node*)); //This queue has to contain all the leafs of the tree
    new_tree->potential_split_nodes->head = 0;
    new_tree->potential_split_nodes->tail=-1;
    new_tree->potential_split_nodes->capacity=pow(2,config->max_depth);
    new_tree->potential_split_nodes->size = 0;
    pushQueue(new_tree->potential_split_nodes, new_tree->root);
    return new_tree;
}

Node* get_potential_split_node(Tree* tree) {
    return popQueue(tree->potential_split_nodes);
}

void getInstancesLabels(double *X, int *pointers, long total_count, long total_left, double **left_instances,
                        int **left_pointers, long total_right, double **right_instances, int **right_pointers,
                        int num_features, const unsigned char *in_left_split, const int *new_pointer_mapping,
                        const double *gh, double** left_gh,double** right_gh,double *left_value,double *right_value) {


    long* left_cur_feature_count = (long*) calloc(num_features, sizeof(long));
    long* right_cur_feature_count = (long*) calloc(num_features, sizeof(long));

    double left_gradients=0;
    double left_hessians=0;
    double right_gradients=0;
    double right_hessians=0;

    //TODO: Going in feature per row may be slower... Reverse order of loops?
    for (long i=0; i<total_count; i++) { // Total_count = num_rows

        for (int f = 0; f < num_features; f++) {
            int orig_row = pointers[f * total_count + i];
            int is_left_split = (in_left_split[orig_row / 8] & (1 << (orig_row % 8))) != 0;

            if (is_left_split) {
                long curr_count = left_cur_feature_count[f];
                (*left_instances)[f * total_left + curr_count] = X[f * total_count + i];
                (*left_pointers)[f * total_left + curr_count] = new_pointer_mapping[orig_row];

                (*left_gh)[2*f * total_left + curr_count] = gh[2*f * total_count + i]; // Gradient
                (*left_gh)[(2*f+1) * total_left + curr_count] = gh[(2*f + 1)* total_count + i]; // Hessian

                left_cur_feature_count[f]++;
            } else {
                long curr_count = right_cur_feature_count[f];
                (*right_instances)[f * total_right + curr_count] = X[f * total_count + i];
                (*right_pointers)[f * total_right + curr_count] = new_pointer_mapping[orig_row];

                (*right_gh)[2*f * total_right + curr_count] = gh[2*f * total_count + i]; // Gradient
                (*right_gh)[(2*f+1) * total_right + curr_count] = gh[(2*f + 1)* total_count + i]; // Hessian

                right_cur_feature_count[f]++;
            }
        }
    }


    for (long i=0; i<total_left; i++) {
        left_gradients+=(*left_gh)[i];
        left_hessians+=(*left_gh)[total_left+i];
    }

    for (long i=0; i<total_right; i++) {
        right_gradients+=(*right_gh)[i];
        right_hessians+=(*right_gh)[total_right+i];
    }

    free(left_cur_feature_count);
    free(right_cur_feature_count);

    *left_value = - left_gradients/(left_hessians + LAMBDA);
    *right_value = - right_gradients/(right_hessians + LAMBDA);


    *left_gh = realloc(*left_gh,(total_left)*2*num_features* sizeof(double));
    *right_gh = realloc(*right_gh,(total_right)*2* num_features* sizeof(double));
    *left_instances = realloc(*left_instances,(total_left)*num_features* sizeof(double));
    *left_pointers = realloc(*left_pointers,(total_left)*num_features* sizeof(int));
    *right_instances = realloc(*right_instances,(total_right)*num_features* sizeof(double));
    *right_pointers = realloc(*right_pointers,(total_right)*num_features* sizeof(int));
}


int *populate_metadata(long total_count, long total_left, long total_right, const unsigned char *in_left_split) {

    // TODO: could speedup if all values have a pointer and change the pointer
    int* new_pointer_mapping = (int*) malloc(total_count * sizeof(int));
    int mapping_left_counter = 0;
    int mapping_right_counter = 0;
    for (long i=0; i<total_count; i++) { // Total_count = num_rows
        int is_left_split = (in_left_split[i / 8] & (1 << (i % 8))) != 0;
        if (is_left_split) {
            new_pointer_mapping[i] = mapping_left_counter;
            mapping_left_counter++;
        } else {
            new_pointer_mapping[i] = mapping_right_counter;
            mapping_right_counter++;
        }
    }

    return new_pointer_mapping;
}

void decide_branch(const double *X, const int *pointers, long total_count, int feature, double val, long *left_count,
                   long *right_count, unsigned char** in_left_split) {
    int total_left = 0;
    int total_right = 0;
    // Bitarray that indicates if a particular row in the original matrix should be in the left / right split
    for (long i=0; i<total_count; i++) {
        double target = X[feature * total_count + i];
        int orig_row = pointers[feature * total_count + i];
        if (target <= val) {
            (*in_left_split)[orig_row/8] |= (1 << (orig_row%8));
            total_left++;
        }
        else {
            (*in_left_split)[orig_row/8] &= ~(1 << (orig_row%8));
            total_right++;
        }
    }

    (*left_count) = total_left;
    (*right_count) = total_right;
}

void split(Tree* tree, Node* node, int feature, double val) {

    node->split_val = val;
    node->split_feature = feature;

    double* X = node->instance_X;
    int* pointers = node->instance_pointers;
    double * gh = node->instance_gh;
    long num_instances = node->count_instances;

    if (num_instances > tree->config->min_instances) {

        long *left_count = (long *) malloc(sizeof(long));
        long *right_count = (long *) malloc(sizeof(long));

        long size = (num_instances / 8) + ((num_instances % 8) != 0);
        unsigned char* in_left_split = (unsigned char*) malloc(size * sizeof(unsigned char));
        // Decides child node for each instance
        decide_branch(X, pointers, num_instances, feature, val, left_count, right_count, (unsigned char **) &in_left_split);

        // Left and right pred
        double* left_value = (double*)malloc(sizeof(double));
        double* right_value = (double*)malloc(sizeof(double));

        double *left_gh = (double *) malloc(2*num_instances * tree->num_features* sizeof(double));
        double *right_gh = (double *) malloc(2*num_instances * tree->num_features* sizeof(double));


        //Responsible for populating grad-hess, and node-value
        int *new_pointer_mapping = populate_metadata(num_instances, *left_count, *right_count, in_left_split);


        // Initialize both with max number of instances they can get
        double *left_instances = (double *) malloc(num_instances * tree->num_features * sizeof(double));
        double *right_instances = (double *) malloc(num_instances * tree->num_features * sizeof(double));

        int *left_pointers = (int *) malloc(num_instances * tree->num_features * sizeof(int));
        int *right_pointers = (int *) malloc(num_instances * tree->num_features * sizeof(int));

        getInstancesLabels(X, pointers, num_instances, *left_count, &left_instances, &left_pointers, *right_count,
                           &right_instances, &right_pointers, tree->num_features, in_left_split, new_pointer_mapping,
                           gh,&left_gh,&right_gh,left_value, right_value);

        Node *left = createNode(*left_value, left_instances, left_pointers, left_gh, *left_count, (node->depth) + 1);
        Node *right = createNode(*right_value, right_instances, right_pointers, right_gh, *right_count, (node->depth) + 1);

        free(left_count);
        free(right_count);

        node->left = left;
        node->right = right;

        pushQueue(tree->potential_split_nodes, left);
        pushQueue(tree->potential_split_nodes, right);

        tree->depth = fmax(tree->depth, left->depth);
    }
}

double node_predict(Node* node, double* X, int row, int n_rows) {
    if (node->left == NULL) { // Leaf
        return node->prediction_val;
    } else {
        if (X[node->split_feature * n_rows + row] <= node->split_val) {
            return node_predict(node->left, X, row, n_rows);
        } else {
            return node_predict(node->right, X, row, n_rows);
        }
    }
}

// X: matrix in column major order containing all rows to predict
// TODO: row major would be more efficient here
void predict(Tree* tree, double* X, int n_rows, double* predictions) {
    for (int r = 0; r < n_rows; r++) {
        predictions[r] = node_predict(tree->root, X, r, n_rows);
    }
}

void print(Node* node) {
    if (node==NULL)
        return;
    for (int i=0; i<node->depth; i++) {
        printf("\t");
    }
    if (node->left == NULL) {
        printf("Instances: %ld, Depth: %d, Log-odds: %f \n",
               node->count_instances, node->depth, node->prediction_val);
    } else {
        printf("Feature: %d, Val: %f, Instances: %ld, Depth: %d, Log-odds: %f \n",node->split_feature, node->split_val,
               node->count_instances, node->depth, node->prediction_val);
    }

    print(node->left);
    print(node->right);
}
