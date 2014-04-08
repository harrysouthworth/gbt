//GBT by Alexandre Michelis

#include "utils.h"
#include "tree.h"



//computes the errors on training and validation sets for the 10 first boosting iterations and each iteration multiple of 100
void gbtEval(double * treematrix, int treevector_size, int interaction_depth, double * nu, double initF, int M, int loss, double * x, double * y, int n_examples, int * inSample_training, int * inSample_valid, double * results) {
   int i,m,j=0;
   double previous_error_training, previous_error_valid, error_training, error_valid;
   double * f = calloc(n_examples, sizeof(double));
   double * predictions = calloc(n_examples, sizeof(double));
   int * inSample = calloc(n_examples, sizeof(int));
   
   for(i=0; i<n_examples; i++) {
      if(inSample_training[i] || inSample_valid[i]) inSample[i] = 1;
   }
   
   for(i=0; i<n_examples; i++) f[i] = initF;
   computeErrorTaV(loss, y, f, inSample_training, inSample_valid, n_examples, &previous_error_training, &previous_error_valid);
   
   for(m=0; m<M; m++) {
      treepartPredict(&treematrix[treevector_size*m], interaction_depth, x, inSample, n_examples, predictions, -1);
      for(i=0; i<n_examples; i++) {
         if(inSample[i]) f[i] += nu[m]*predictions[i];
      }
      if(m < 10 || (m+1)%100 == 0) {
         computeErrorTaV(loss, y, f, inSample_training, inSample_valid, n_examples, &error_training, &error_valid);
         results[5*j] = m+1;
         results[5*j+1] = error_training;
         results[5*j+2] = nu[m];
         results[5*j+3] = error_valid;
         results[5*(j++)+4] = previous_error_valid-error_valid;
      }
      if(m < 9 || (m+2)%100 == 0) {
         computeErrorTaV(loss, y, f, inSample_training, inSample_valid, n_examples, &previous_error_training, &previous_error_valid);
      }
   }
   
   free(f);
   free(predictions);
   free(inSample);
}


//for conjugate gradient: computes the new descent direction r according to the Polak-Ribière coefficient that it also computes
double betaPR(double * previous_negative_gradient, double * negative_gradient, int n, double * previous_r, double * r) {
   int i;
   double num=0, denom=0, tmp;

//#pragma omp parallel for reduction(+:num) reduction(+:denom) schedule(static)
   for(i=0; i<n; i++) {
      num += (negative_gradient[i]-previous_negative_gradient[i])*negative_gradient[i];
      denom += previous_negative_gradient[i]*previous_negative_gradient[i];
   }
   tmp = max(0,num/denom);
//#pragma omp parallel for schedule(static)
   for(i=0; i<n; i++) {
      r[i] = negative_gradient[i] +  tmp * previous_r[i];
   }
   
   return(num/denom);
}


//initializes the parameters that will be used to compute the nu (shrinkage) values
void init_nu(SEXP s_shrinkage, int n_trees, int * shrinkage_type, double * nu_a, double * nu_b, double * nu_c) {
   //0 for fixed, 1 for arithmetic, 2 for geometric, 3 for negative exponential
   *shrinkage_type = *INTEGER(getListElement(s_shrinkage, "type"));
   if(*shrinkage_type == 0) {
      *nu_a = *REAL(getListElement(s_shrinkage, "value"));
   }
   else if(*shrinkage_type == 1) {
      *nu_a = *REAL(getListElement(s_shrinkage, "start"));
      *nu_b = ((*REAL(getListElement(s_shrinkage, "end"))) - (*nu_a))/n_trees;
   }
   else if(*shrinkage_type == 2) {
      *nu_a = *REAL(getListElement(s_shrinkage, "start"));
      *nu_b = R_pow(n_trees, log((*REAL(getListElement(s_shrinkage, "end")))/(*nu_a))/(n_trees * log(n_trees)));
   }
   else if(*shrinkage_type == 3) {
      *nu_a = log(0.25)/(1-(*REAL(getListElement(s_shrinkage, "iter75"))));
      *nu_b = ((*REAL(getListElement(s_shrinkage, "start")))-(*REAL(getListElement(s_shrinkage, "end"))))/(exp(-(*nu_a))-exp(-(*nu_a) * n_trees));
      *nu_c = (*REAL(getListElement(s_shrinkage, "end")))-(*nu_b) * exp(-(*nu_a) * n_trees);
   }
}

//computes the new shrinkage (step-size) value
double update_nu(int n_trees, int * shrinkage_type, double * nu_a, double * nu_b, double * nu_c, int m) {
   //0 for fixed, 1 for arithmetic, 2 for geometric, 3 for negative exponential
   if(*shrinkage_type == 0) {
      return(*nu_a);
   }
   else if(*shrinkage_type == 1) {
      return((*nu_a) + m * (*nu_b));
   }
   else if(*shrinkage_type == 2) {
      return((*nu_a) * R_pow(*nu_b, m));
   }
   else if(*shrinkage_type == 3) {
      return((*nu_b) * exp(-(*nu_a) * m) + (*nu_c));
   }
   return 0;
}











/*
inputs:
- s_loss: 0 for squared loss, 1 for binomial deviance
- s_n_trees: the number of trees to fit
- s_n_features: an int, the number of features
- s_training_size: an int, the number of examples to use for training
- s_valid_size: an int, the number of examples in the validation set (they follow the training set)
- s_x: the data, a vector of (s_n_features+1)*(s_training_size+s_valid_size) doubles (the first (s_training_size+s_valid_size) doubles correspond to the intercept, not a real feature...)
- s_y: the response, a vector of (s_training_size+s_valid_size) doubles
- s_ordered_indexes: an int vector organized just like s_x, containing the ordered indexes according to each feature
- s_training_indexes: an int vector of size s_training_size containing the indexes of the training set examples (starts from 1 or more (R indexes...))
- s_inSample_training: an int vector of 0 (example not in training set) and s_training_size 1 (example in training set)
- s_inSample_valid an int vector of 0 (example not in valid set) and s_valid_size 1 (example in valid set)
- s_feature_type: a char* containing the type of each feature (B-inary, M-onovalue (useless), R-eal)
- s_split_value: a double vector containing the known split value for each feature (correct only for binary features)
- s_sample_size: an int, the number of example to use at each iteration
- s_shrinkage: a list depending on the type of shrinkage (which contains the element named "type": 0 for fixed, 1 for arithmetic, 2 for geometric, 3 for negative exponential)
- s_initF: a double, the constant prediction
- s_conjugateGradient: an int, 1 to use a conjugate gradient method, 0 for a regular gradient descent
- s_interaction_depth: an int, max depth of each tree
- s_treevector_size: an int, number of elements that make a treevector
- s_compute_results: 1 to compute the results, 0 otherwise
outputs:
- s_treematrix: a vector of s_n_trees*s_treevector_size doubles used to store the trees
- s_nu: a vector of s_n_trees doubles to store the nu values
- s_gamma: a vector of s_n_trees doubles to store the gamma values
- s_results: a vector of "5 columns times the correct number of rows" doubles to contain 5 values (iteration number, training error, step size (nu), test error, improve) per iteration to store (1 to 10 and each multiple of 100)
*/
SEXP gbt(SEXP s_loss, SEXP s_n_trees, SEXP s_n_features, SEXP s_training_size, SEXP s_valid_size, SEXP s_x, SEXP s_y, SEXP s_ordered_indexes, SEXP s_training_indexes, SEXP s_inSample_training, SEXP s_inSample_valid, SEXP s_feature_type, SEXP s_split_value, SEXP s_sample_size, SEXP s_shrinkage,   SEXP s_initF, SEXP s_conjugateGradient, SEXP s_interaction_depth, SEXP s_treevector_size, SEXP s_compute_results, SEXP s_treematrix, SEXP s_nu, SEXP s_results) {
   //inputs
   int loss = *INTEGER(s_loss);
   int n_trees = *INTEGER(s_n_trees);
   int n_features = *INTEGER(s_n_features);
   int training_size = *INTEGER(s_training_size);
   int valid_size = *INTEGER(s_valid_size);
   double * x = REAL(s_x);
   double * y = REAL(s_y);
   int * ordered_indexes = INTEGER(s_ordered_indexes);
   int * training_indexes = INTEGER(s_training_indexes);
   int * inSample_training = INTEGER(s_inSample_training);
   int * inSample_valid = INTEGER(s_inSample_valid);
   char * feature_type = (char*)CHAR(STRING_ELT(s_feature_type, 0));
   double * split_value = REAL(s_split_value);
   int sample_size = *INTEGER(s_sample_size);
   int shrinkage_type;
   double initF = *REAL(s_initF);
   int conjugateGradient = *INTEGER(s_conjugateGradient);
   int interaction_depth = *INTEGER(s_interaction_depth);
   int treevector_size = *INTEGER(s_treevector_size);
   int compute_results = *INTEGER(s_compute_results);
   //outputs
   double * treematrix = REAL(s_treematrix);
   double * nu = REAL(s_nu);
   double * results = REAL(s_results);
   //others
   double nu_a, nu_b, nu_c, beta_PR;
   int n_examples = training_size+valid_size;
   int * inSample_sampled = (int*)calloc(n_examples, sizeof(int));
   double * r = (double*)malloc(n_examples*sizeof(double));
   double * previous_r;
   double * negative_gradient, * previous_negative_gradient, * previous_F, *predictions;
   int i,m;
   
   predictions = (double *)malloc(n_examples*sizeof(double));
   
   //first approximation: constant value
   previous_F = (double *)malloc(n_examples*sizeof(double));
   for(i=0; i<n_examples; i++) {
      if(inSample_training[i]) {
         previous_F[i] = initF;
      }
   }
   
   //compute gradient if needed
   if(conjugateGradient) {
      negative_gradient = (double*)malloc(n_examples*sizeof(double));
      previous_negative_gradient = (double*)malloc(n_examples*sizeof(double));
      previous_r = (double*)calloc(n_examples,sizeof(double));
      negativeGradient(loss, n_examples, inSample_training, y, previous_F, previous_negative_gradient);
   }
   GetRNGstate();
   
   init_nu(s_shrinkage, n_trees, &shrinkage_type, &nu_a, &nu_b, &nu_c);
   
   
   //boosting iterations
   for(m=0; m<n_trees; m++) {
      //create training sample (assume inSample_sampled was 0 for all examples)
      newSample(sample_size, training_size, inSample_sampled, training_indexes);
      
      //compute descent direction
      if(conjugateGradient) {
         negativeGradient(loss, n_examples, inSample_training, y, previous_F, negative_gradient);
         beta_PR = betaPR(previous_negative_gradient, negative_gradient, n_examples, previous_r, r);
         // if(beta_PR < 0) {
            // Rprintf("m %d, beta_PR %f\n", m, beta_PR);
         // }
         memcpy(previous_negative_gradient, negative_gradient, n_examples*sizeof(double));
         memcpy(previous_r, r, n_examples*sizeof(double));
      }
      else {
         negativeGradient(loss, n_examples, inSample_sampled, y, previous_F, r);
      }
      
      //fit a base learner to the descent direction
      rec_regTree(loss, r, x, y, ordered_indexes, n_examples, interaction_depth, 0, inSample_sampled, sample_size, n_features, feature_type, split_value, &treematrix[treevector_size*m]);
      
      //predict (on the entire training set as we'll need it later)
      treepartPredict(&treematrix[treevector_size*m], interaction_depth, x, inSample_training, n_examples, predictions, -1);
      
      
      //shrinkage
      nu[m] = update_nu(n_trees, &shrinkage_type, &nu_a, &nu_b, &nu_c, m);
      
      //reset sample
      for(i=0; i<n_examples; i++) inSample_sampled[i] = 0;
      //update the predictions on the entire training set
      for(i=0; i<n_examples; i++) {
         if(inSample_training[i]) {
            previous_F[i] +=  nu[m]*predictions[i];
         }
      }
   }
   
   if(compute_results) {
      gbtEval(treematrix, treevector_size, interaction_depth, nu, initF, n_trees, loss, x, y, n_examples, inSample_training, inSample_valid, results);
   }
   
   PutRNGstate();
   free(inSample_sampled);
   free(r);
   if(conjugateGradient) {
      free(negative_gradient);
      free(previous_negative_gradient);
      free(previous_r);
   }
   free(previous_F);
   free(predictions);
   
   return R_NilValue;
}







void rec_ri(double * ri, double previous_ss, double * treevector, int k, int max_k) {
   int index=6*k;
   double error;
   if(k >= max_k) return;
   if(treevector[index] > 0) {
      error = treevector[index+2] + treevector[index+3];
      ri[(int)treevector[index]] += (previous_ss - error)*(previous_ss - error);
      rec_ri(ri, treevector[index+2], treevector, 2*k, max_k);
      rec_ri(ri, treevector[index+3], treevector, 2*k+1, max_k);
   }
}

//output: double * ri
SEXP ri(SEXP s_treematrix, SEXP s_ri, SEXP s_treevector_size, SEXP s_M, SEXP s_depth) {
   double * treematrix = REAL(s_treematrix);
   double * ri = REAL(s_ri);
   int treevector_size = *INTEGER(s_treevector_size);
   int M = *INTEGER(s_M);
   int depth = *INTEGER(s_depth);
   int max_k = (int)R_pow(2,depth);
   
   int m;
   
   for(m=0; m<M; m++) {
      rec_ri(ri, treematrix[treevector_size*m+1], &treematrix[treevector_size*m], 1, max_k);
   }
   
   return(R_NilValue);
}





//PREDICT

SEXP predict(SEXP s_treematrix, SEXP s_nu, SEXP s_x, SEXP s_n_examples, SEXP s_n_trees, SEXP s_treevector_size, SEXP s_depth, SEXP s_predictions, SEXP s_initF) {
   double * treematrix = REAL(s_treematrix);
   double * nu = REAL(s_nu);
   double * predictions = REAL(s_predictions);
   double * x = REAL(s_x);
   double initF = *REAL(s_initF);
   int treevector_size = *INTEGER(s_treevector_size);
   int M = *INTEGER(s_n_trees);
   int n_examples = *INTEGER(s_n_examples);
   int depth = *INTEGER(s_depth);
   int* inSample = (int*)malloc(n_examples * sizeof(int));
   double * f = calloc(n_examples, sizeof(double));
   
   int m,i;
   for(m=0; m<n_examples; m++) {
      inSample[m] = 1;
      predictions[m] = initF;
   }
   
   
   for(m=0; m<M; m++) {
      treepartPredict(&treematrix[treevector_size*m], depth, x, inSample, n_examples, f, -1);
      for(i=0; i<n_examples; i++) {
         predictions[i] += nu[m]*f[i];
      }
   }
   free(inSample);
   free(f);
   return(R_NilValue);
}
