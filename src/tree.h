//GBT by Alexandre Michelis

#ifndef TREE_H
#define TREE_H

#include "utils.h"

double residualsLoss(int loss, double y, double f);

//according to the loss, y and f, it computes the negative gradient for the allowed rows up to the first size elements
void negativeGradient(int loss, int size, int * inSample, double * y, double * f, double * negative_gradient);

//compute the error on the sample
double computeError(int loss, double * y, double * f, double stepSize, double * predictions, int * inSample, int size);

//compute the errors on the training and valid sets
void computeErrorTaV(int loss, double * y, double * f, int * inSample_training, int * inSample_valid, int size, double * error_training, double * error_valid);


void treepartPredict(double * treevector, int depth, double * features, int * inSample, int nb_examples, double * predictions, int n);

/*
outputs:
treevector:
   - vector of doubles
   - each tree node is represented by 6 doubles: feature number, split value, error, error left, error_right, left value, right value
   - nodes are numbered as follow:
      - root = 0
      - first node = 1
      - left child of node k is 2*k
      - right child of node k is 2*k+1
   - node k is located in s_treevector at index 6*k
*/
void rec_regTree(int loss, double * residuals, double * data, double * response, int * ordered_indexes, int nb_examples, int depth_remaining, int k, int * inSample, int nb_allowed, int nb_features, char * feature_type, double * split_value, double * treevector);

#endif