//
// Created by Rishabh Singh on 14/04/2020.
//

#include "inputreader.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define BUFFER_SIZE 4098
#define MAX(a,b) (((a)>(b))?(a):(b))

/**
 * Extracts the amount of rows, columns and the max line width (in characters) from a CSV file
 * @param path path to the file to extract the information from
 * @param rows pointer to write the amount of rows to
 * @param columns pointer to write the amount of columns to
 * @param max_width pointer to write the max width to
 * @return 0 if the extraction succeeded
 */
int getRowsColumnsMaxWidth(char *path, int* rows, int* columns, int* max_width) {
    FILE *f = fopen(path, "r" );
    if (f == NULL) return -1;

    // Extract amount of rows in csv
    int n_rows = 0;

    char buffer[BUFFER_SIZE];
    long line_width = 0;
    int max_line_width = 0;
    int extra = 0;

    while (fgets(buffer, sizeof(buffer), f) != NULL) {
        char* pos = strchr(buffer,'\n');
        if (pos) {
            max_line_width = MAX(max_line_width, line_width + pos-buffer+1);
            line_width = 0;
            n_rows++;
            extra = 0;
        } else {
            line_width += BUFFER_SIZE-1;
            extra = 1;
        }
    }
    max_line_width = MAX(max_line_width, line_width);
    n_rows += extra;

    // Extract amount of columns in csv
    int n_cols = 1;

    rewind(f);
    char line[max_line_width+2];
    fgets(line, sizeof(line), f);
    for (int i = 0; i < max_line_width; i++) {
        if (line[i] == ',') {
            n_cols ++;
        }
    }

    fclose(f);

    *rows = n_rows;
    *columns = n_cols;
    *max_width = max_line_width;

    return 0;
}

/**
 * Reads the file at the given path into an Input struct
 * @param path the path to read the file from
 * @param label_col column in the csv file which contains the labels
 * @param n_features amount of features
 * @param feature_cols columns containing the features
 * @param result pointer to an input struct to write the parsed data to
 * @return 0 if the file reading succeeded
 */
int read_csv(char* path, int label_col, int n_features, int* feature_cols, struct Input* result) {

    int nrow; int ncol; int max_line_width;
    getRowsColumnsMaxWidth(path, &nrow, &ncol, &max_line_width);

    FILE *fstream = fopen(path,"r");
    if (!fstream) return -1;

    double* X = (double *)malloc(nrow*n_features*sizeof(double));
    double* Y = (double*)malloc(nrow*sizeof(double));

    // Bit array that indicates which columns are used as features
    unsigned char is_feature_col[(ncol / 8) + ((ncol % 8) != 0)];

    // Initialize to 0
    for (int i = 0; i < ncol; i++) {
        is_feature_col[i/8] &= ~(1 << (i%8));
    }

    // Set col bits to 1
    for (int i = 0; i < n_features; i++) {

        int col = feature_cols[i];
        is_feature_col[col/8] |= (1 << (col%8));
    }

    // Read CSV file into 2D array
    int cur_row = 0, cur_col = 0;
    int i = 0, j = 0;

    char *record,*line;
    char buffer[max_line_width+2];

    int total_rows = nrow - 1; // Don't include header

    while((line=fgets(buffer,sizeof(buffer),fstream)) != NULL && i < nrow + 1) {
        if(i != 0) { // Skip header
            record = strtok(line,",");
            j = 0, cur_col = 0;

            while(record != NULL) {
                int is_feature = (is_feature_col[j/8] & (1 << (j%8))) != 0;
                int is_label = j == label_col;

                if (is_label) {
                    Y[cur_row] = (double) atof(record);
                } else if (is_feature) {
                    X[total_rows * cur_col + cur_row] = (double) atof(record);
                    cur_col ++;
                }

                record = strtok(NULL, ",");
                j++;
            }
            cur_row++;
        }
        i++;
    }

    fclose(fstream);

    result->X = X;
    result->Y = Y;
    result->n_features = n_features;
    result->n_rows = total_rows;

    return 0;
}