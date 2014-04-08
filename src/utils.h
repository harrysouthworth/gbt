//GBT by Alexandre Michelis
#ifndef UTILS_H
#define UTILS_H

#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>

//returns the element of name str from the list list
SEXP getListElement(SEXP list, const char *str);

//puts value 1 in k elements of inSample, according to the k randomly picked values of indexes
//indexes must be an int vector of size n containing exactly each allowed index once
void newSample(int k, int n, int * inSample, int *indexes);

//returns the first element in the sample (starting from index start included) up to n_rows
int nextAllowedIndex(int start, int* ordered_indexes, int* inSample, int n_rows);

//returns the max of a and b
double max(double a, double b);

#endif