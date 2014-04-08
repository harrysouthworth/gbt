//GBT by Alexandre Michelis
#include "utils.h"

SEXP getListElement(SEXP list, const char *str) {
   SEXP elmt = R_NilValue, names = getAttrib(list, R_NamesSymbol);

   for (R_len_t i = 0; i < length(list); i++)
       if(strcmp(CHAR(STRING_ELT(names, i)), str) == 0) {
          elmt = VECTOR_ELT(list, i);
          break;
       }
   return elmt;
}

void newSample(int k, int n, int * inSample, int *indexes) {
   int i, j, tmp;
   for (i = 0; i < k; i++) {
      j = n * unif_rand();
      inSample[indexes[j]] = 1;
      tmp = indexes[j];
      indexes[j] = indexes[--n];
      indexes[n] = tmp;
   }
}


R_INLINE int nextAllowedIndex(int start, int* ordered_indexes, int* inSample, int n_rows) {
   int i = start;
   for(; i<n_rows; i++) {
      if(inSample[ordered_indexes[i]] == 1) return i;
   }
   return (n_rows+1);
}


R_INLINE double max(double a, double b) {
   if(a>b) return a;
   return b;
}