//GBT by Alexandre Michelis

#include "tree.h"



//first this file includes all functions relative to the loss functions



R_INLINE double residualsLoss(int loss, double y, double f) {
   double tmp;
   if(loss == 0) { //squared loss
      return((y-f)*(y-f));
   }
   else { //binomial deviance
      tmp = exp(f);
      if(isinf(tmp)) tmp = DBL_MAX; //exponential too big
      tmp = log(1 + tmp)-y*f;
      if(isinf(tmp)) return(DBL_MAX);
      return(tmp);
   }
}



R_INLINE double negativeGradientLoss(int loss, double y, double f) {
   if(loss == 0) { //squared loss
      return(y-f);
   }
   else { //binomial deviance
      return(y-1/(1+exp(-f)));
   }
}






void negativeGradient(int loss, int size, int * inSample, double * y, double * f, double * negative_gradient) {
   int i;
   for(i=0; i<size; i++) {
      if(inSample[i]) {
         negative_gradient[i] = negativeGradientLoss(loss, y[i], f[i]);
      }
   }
}



void computeErrorTaV(int loss, double * y, double * f, int * inSample_training, int * inSample_valid, int size, double * error_training, double * error_valid) {
   int i,n_t=0,n_v=0;
   *error_training = 0;
   *error_valid = 0;
   
   for(i=0; i<size; i++) {
      if(inSample_training[i] || inSample_valid[i]) {
         if(inSample_training[i]) { //if training sample
            (*error_training) += residualsLoss(loss, y[i], f[i]);
            n_t++;
         }
         else { //if valid sample
            (*error_valid) += residualsLoss(loss, y[i], f[i]);
            n_v++;
         }
      }
   }
   
   if(loss == 0) { //squared loss
      (*error_training) /= (2*max(n_t,1));
      (*error_valid) /= (2*max(n_v,1));
   }
   else { //binomial deviance
      (*error_training) *= 2/max(n_t,1);
      (*error_valid) *= 2/max(n_v,1);
   }
}








//compute the initial total sum, sum of squares and whatever other value S3T is required for the loss function when looking for the splits
void initSplits(int loss, double * residuals, double * original_response, int * inSample, int nb_examples, double * ST, double * S2T, double * S3T) {
   int i;
   *ST = 0;
   *S2T = 0;
   *S3T = 0;
   
   for(i=0; i<nb_examples; i++) {
      if(inSample[i] == 1) {
         (*ST) += residuals[i];
         (*S2T) += residuals[i]*residuals[i];
         if(loss == 1) {
            (*S3T) += (original_response[i] - residuals[i])*(1 - original_response[i] + residuals[i]);
         }
      }
   }
}



//maintains up to 3 values for the left (L1, L2, L3) and 3 for the right (R1, R2, R3) of the split when y goes to the left part of the split
//in order to be able to compute the best splits/constants
//these values are maintained differently according to the loss function used
R_INLINE void maintainSplit(int loss, double residuals, double original_response, int * nbL, int * nbR, double * SL, double * S2L, double * L3, double * SR, double * S2R, double * R3) {
   //S = sums
   //S2 = sums of squares
   (*SL) += residuals;
   (*S2L) += residuals*residuals;
   (*SR) -= residuals;
   (*S2R) -= residuals*residuals;
   (*nbL)++;
   (*nbR)--;
   if(loss == 1) { //binomial deviance
      (*L3) += (original_response - residuals)*(1 - original_response + residuals);
      (*R3) -= (original_response - residuals)*(1 - original_response + residuals);
   }
}

//compute the best contants for the split according to the loss function
R_INLINE void bestConstants(int loss, int nbL, int nbR, double SL, double S2L, double L3, double SR, double S2R, double R3, double * CL, double * CR) {
   if(loss == 0) { //squared loss
      *CL = SL/nbL;
      *CR = SR/nbR;
   }
   else { //binomial deviance
      *CL = SL/L3;
      *CR = SR/R3;
   }
}







/*
Outputs:
split_value the chosen split value
ss the sum of squares that is induced by this split
CL the constant of the left part of the split
CR the constant of the right part of the split
*/
int splitFeature(int loss, double * residuals, double * response, char type, double value, int * ordered_indexes, double * feature, int * inSample, int nb_allowed, double ST, double S2T, double S3, double * split_value, double * ss, double * ss_L, double * ss_R, double * CL, double * CR, int n_rows, int numf, int k) {
   double SL=0, SR=0; //left and right sum(y)
   double S2L=0, S2R=0; //left and right sum(y^2)
   double L3=0, R3=0;
   int nbL, nbR;
   double L, R, current_ss, previous_el_feature_val; //ss left and right, and global
   int tmp, j, index_previous_val, previous_el_index, first_split=1;
   
   if(nb_allowed < 2) {
      return 0;
   }
   if(type == 'M') { //monovalue feature (useless)
      return 0;
   }
   else if(type == 'B') { //binary feature (we know where the split is)
      j = -1;
      
      nbL = 0; //number of values to the left
      nbR = nb_allowed; //number of values to the right
      SL = 0;
      SR = ST;
      S2L = 0;
      S2R = S2T;
      L3 = 0;
      R3 = S3;
      *CL = 0;
      *CR = 0;
      do {
         j = nextAllowedIndex(j+1, ordered_indexes, inSample, n_rows);
         if(j <= n_rows) {
            tmp = ordered_indexes[j];
            if(feature[tmp] <= value) { //goes to the left
               maintainSplit(loss, residuals[tmp], response[tmp], &nbL, &nbR, &SL, &S2L, &L3, &SR, &S2R, &R3);
            }
            else {
               break; // we reached the split point
            }
         }
      } while(j<=n_rows);
      if(nbL==0 || nbR==0) { //useless split
         return 0;
      }
      *ss = S2T - SL*SL/nbL - SR*SR/nbR;
      *ss_L = S2L - SL*SL/nbL;
      *ss_R = S2R - SR*SR/nbR;
      bestConstants(loss, nbL, nbR, SL, S2L, L3, SR, S2R, R3, CL, CR);
      *split_value = value;
   }
   else {
      nbL = 0; //number of values to the left
      nbR = nb_allowed; //number of values to the right
      SL = 0;
      SR = ST;
      S2L = 0;
      S2R = S2T;
      L3 = 0;
      R3 = S3;
      *CL = 0;
      *CR = 0;
   
   
      //we start with one element to the left
      j = nextAllowedIndex(0, ordered_indexes, inSample, n_rows);
      tmp = ordered_indexes[j];
      
      maintainSplit(loss, residuals[tmp], response[tmp], &nbL, &nbR, &SL, &S2L, &L3, &SR, &S2R, &R3);
      
      L = 0; //only one value => perfect fit
      R = S2R - SR*SR/nbR;
      current_ss = L+R;
      
      //we have our starting values
      index_previous_val = 0;
      previous_el_index = 0;
      previous_el_feature_val = feature[tmp];
      *split_value = previous_el_feature_val; //we don't really have a split yet, but we need it in case we find no split until the end (to check)
      bestConstants(loss, nbL, nbR, SL, S2L, L3, SR, S2R, R3, CL, CR);
      
      tmp = 0;
      //slide elements to the left part one by one
      while(nbR > 0) {
         j = nextAllowedIndex(j+1, ordered_indexes, inSample, n_rows);
         tmp = ordered_indexes[j];
         
         
         if(previous_el_feature_val != feature[tmp]) { //new complete split
            if(current_ss < *ss || first_split) { //new best split
               first_split = 0;
               *ss = current_ss;
               *ss_L = S2L - SL*SL/nbL;
               *ss_R = S2R - SR*SR/nbR;
               bestConstants(loss, nbL, nbR, SL, S2L, L3, SR, S2R, R3, CL, CR);
               *split_value = previous_el_feature_val;
               index_previous_val = previous_el_index;
            }
         }
         
         //update values
         maintainSplit(loss, residuals[tmp], response[tmp], &nbL, &nbR, &SL, &S2L, &L3, &SR, &S2R, &R3);
         L = S2L - SL*SL/nbL;
         R = S2R - SR*SR/nbR;
         current_ss = L+R;
         previous_el_feature_val = feature[tmp];
         previous_el_index = j;
      }
      
      //if no good split
      if(*CL == *CR || *split_value == previous_el_feature_val) {
         return 0;
      }
      
      *split_value = (*split_value+feature[ordered_indexes[nextAllowedIndex(index_previous_val+1, ordered_indexes, inSample, n_rows)]])/2; //true split value
   }
   
   return 1;
}







void rec_regTree(int loss, double * residuals, double * data, double * response, int * ordered_indexes, int nb_examples, int depth_remaining, int k, int * inSample, int nb_allowed, int nb_features, char * feature_type, double * split_value, double * treevector) {
   int i,f;
   double ST=0, S2T=0, S3T=0; //sum, sum of squares (of the residuals) and whatever is needed by the loss function
   double res_split_value, res_ss, res_CL, res_CR, ss_L, ss_R; //values returned for the current feature
   double best_split_value, best_ss=0, best_CL, best_CR, best_ss_L, best_ss_R; //best values returned so far
   int best_feature=-1,got_a_split=0,number_of_splits=0; //best feature so far
   int * inSample_left, * inSample_right;
   int nb_allowed_left=nb_allowed, nb_allowed_right=nb_allowed;
   
   initSplits(loss, residuals, response, inSample, nb_examples, &ST, &S2T, &S3T);
   
   if(k==0) {//root
      bestConstants(loss, nb_allowed, 0, ST, S2T, S3T, 0, 0, 0, &treevector[0], &res_split_value); //best constant
      //error with this prediction
      treevector[1] = 0;
      for(i=0; i<nb_examples; i++) {
         if(inSample[i]) {
            treevector[1] += residualsLoss(loss, residuals[i], treevector[0]);
         }
      }
      treevector[2] = -1;
      treevector[3] = -1;
      treevector[4] = -1;
      treevector[5] = -1;
      rec_regTree(loss, residuals, data, response, ordered_indexes, nb_examples, depth_remaining, 1, inSample, nb_allowed, nb_features, feature_type, split_value, treevector);
   }
   else {
      //f starts at 1 as the first column (number 0) isn't a feature
#pragma omp parallel for schedule(dynamic) private(i,got_a_split,res_split_value,res_ss,ss_L,ss_R,res_CL,res_CR)
      for(f=1; f<=nb_features; f++) {
         i = nb_examples*f; //start index for this feature
         got_a_split = splitFeature(loss, residuals, response, feature_type[f-1], split_value[f-1], &ordered_indexes[i], &data[i], inSample, nb_allowed, ST, S2T, S3T, &res_split_value, &res_ss, &ss_L, &ss_R, &res_CL, &res_CR, nb_examples,f,k);
#pragma omp critical
{
         number_of_splits += got_a_split;
         if(got_a_split==1 && (res_ss < best_ss || number_of_splits == 1)) {
            best_feature = f;
            best_split_value = res_split_value;
            best_ss = res_ss;
            best_ss_L = ss_L;
            best_ss_R = ss_R;
            best_CL = res_CL;
            best_CR = res_CR;
         }
}
      }
      
      treevector[6*k] = -1;
      //if we got our best split
      if(number_of_splits != 0) {
         //store the values
         treevector[6*k] = best_feature;
         treevector[6*k+1] = best_split_value;
         treevector[6*k+2] = 0;
         treevector[6*k+3] = 0;
         treevector[6*k+4] = best_CL;
         treevector[6*k+5] = best_CR;
         
         
         //compute allowed rows and errors
         inSample_left = (int *)malloc(nb_examples*sizeof(int));
         inSample_right = (int *)malloc(nb_examples*sizeof(int));
         memcpy(inSample_left, inSample, nb_examples*sizeof(int));
         memcpy(inSample_right, inSample, nb_examples*sizeof(int));
         
         for(i=0; i<nb_examples; i++) {
            if(inSample[i] == 1) {
               if(data[nb_examples*best_feature+i] <= best_split_value) { //goes to the left
                  inSample_right[i] = 0;
                  nb_allowed_right--;
                  treevector[6*k+2] += residualsLoss(loss, residuals[i], treevector[6*k+4]); //left error
               }
               else {
                  inSample_left[i] = 0;
                  nb_allowed_left--;
                  treevector[6*k+3] += residualsLoss(loss, residuals[i], treevector[6*k+5]); //right error
               }
            }
         }
         
         depth_remaining--;
         if(depth_remaining > 0) {
            //left child
            rec_regTree(loss, residuals, data, response, ordered_indexes, nb_examples, depth_remaining, 2*k, inSample_left, nb_allowed_left, nb_features, feature_type, split_value, treevector);
            
            //right child
            rec_regTree(loss, residuals, data, response, ordered_indexes, nb_examples, depth_remaining, 2*k+1, inSample_right, nb_allowed_right, nb_features, feature_type, split_value, treevector);
         }
         free(inSample_left);
         free(inSample_right);
      }
      else {
         treevector[6*k+1] = -1;
         treevector[6*k+2] = -1;
         treevector[6*k+3] = -1;
         treevector[6*k+4] = -1;
         treevector[6*k+5] = -1;
      }
   }
}











void treepartPredict(double * treevector, int depth, double * features, int * inSample, int nb_examples, double * predictions, int n) {
   int i,j=0,k,n2,index,max_k=(int)R_pow(2,depth);
   
   if(n < 0) { //the indexes of predictions correspond to the ones of inSample
      n2 = nb_examples;
   }
   else { //they don't, predictions is a contiguous vector that will contain the result
      n2 = n;
   }
   
   for(i=0; i<nb_examples && j<n2; i++) {
      if(inSample[i] == 1) {
         if(n<0) {
            j = i;
         }
         predictions[j] = treevector[0];
         k = 1;
         index = 6;
         while(k<max_k) { //we suppose there is at least a root node
            if(features[(int)treevector[index]*nb_examples+i] <= treevector[index+1]) { //left part
               k = 2*k;
               if(k < max_k && treevector[6*k] > 0) { //there's a left child
                  index = 6*k; //go check it
               }
               else { //no left child, get the left value
                  predictions[j] = treevector[index+4];
                  break;
               }
            }
            else { //right part
               k = 2*k+1;
               if(k < max_k && treevector[6*k] > 0) { //there's a right child
                  index = 6*k; //go check it
               }
               else { //no right child, get the right value
                  predictions[j] = treevector[index+5];
                  break;
               }
            }
         }
         if(n>=0) {
            j++;
         }
      }
   }
}

