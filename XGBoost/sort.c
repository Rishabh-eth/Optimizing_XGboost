//
// Created by Noman Sheikh on 13/04/2020.
//

#include "sort.h"
#include <stdlib.h>

// TODO: adjust for column major
/*void bucketSort(int* arr, double* X, int N, int feature, int N_COLS) {
    // Find Minimum and Maximum
    double minElem = 100000, maxElem = -100000;
    for (int i=0; i<N; i++) {
        double val = X[i*N_COLS + feature];
        if (val < minElem) {
            minElem = val;
        }
        if (val > maxElem) {
            maxElem = val;
        }
    }
    int INIT_CAP = 8;
    int minBucket = (int)minElem;
    int maxBucket = (int)maxElem;
    int M = maxBucket - minBucket + 1;
    int* bucket_freq = (int*) malloc(M * sizeof(int));
    int* bucket_cap = (int*) malloc(M * sizeof(int));
    int** bucket_elems = (int**) malloc( M * sizeof(int*));
    for (int i=0; i<M; i++) {
        bucket_freq[i] = 0;
        bucket_cap[i] = INIT_CAP;
        bucket_elems[i] = (int*) malloc(N * sizeof(int));
    }
    for (int i=0; i<N; i++) {
        int val = (int)X[i*N_COLS + feature] - minBucket;
        int count = bucket_freq[val];
        if (count == bucket_cap[val]) {
            bucket_elems[val] = realloc(bucket_elems[val], 2*bucket_cap[val]*sizeof(int));
            bucket_cap[val] *= 2;
        }
        bucket_elems[val][count] = i;
        bucket_freq[val]++;
    }

    int offset = 0;
    for (int i=0; i<M; i++) {
        mergeSort(bucket_elems[i], X, 0, bucket_freq[i]-1, feature, N_COLS);
        for (int j=0; j<bucket_freq[i]; j++) {
            arr[offset+j] = bucket_elems[i][j];
        }
        offset += bucket_freq[i];
        free(bucket_elems[i]);
    }
    free(bucket_freq);
    free(bucket_cap);
}*/

void merge(int* arr, const double* X, int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 =  r - m;

    /* create temp arrays */
    int* L = malloc(n1 * sizeof(int));
    int* R = malloc(n2 * sizeof(int));

    /* Copy data to temp arrays L[] and R[] */
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1+ j];

    /* Merge the temp arrays back into arr[l..r]*/
    i = 0; // Initial index of first subarray
    j = 0; // Initial index of second subarray
    k = l; // Initial index of merged subarray
    while (i < n1 && j < n2) {
        if (X[L[i]] <= X[R[j]]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    /* Copy the remaining elements of L[], if there are any */
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    /* Copy the remaining elements of R[], if there are any */
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
    free(L);
    free(R);
}

/* l is for left index and r is right index of the
   sub-array of arr to be sorted */
void mergeSort(int* arr, double* X, int l, int r)
{
    if (l < r)
    {
        int m = l+(r-l)/2;
        // Sort first and second halves
        mergeSort(arr, X, l, m);
        mergeSort(arr, X, m+1, r);
        merge(arr, X, l, m, r);
    }
}

void sorted_data(double* X, int N, int* arr) {
    for(int i=0; i<N; i++) {
        arr[i] = i;
    }
    mergeSort(arr, X, 0, N-1);
}