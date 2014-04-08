gbt.fit <-
function( formula,
                     loss="squaredLoss",
                     data,
                     n.trees,
                     interaction.depth,
                     shrinkage,
                     bag.fraction,
                     cv.folds,
                     conjugate.gradient,
                     store.results,
                     verbose)
{
   gbt.defaultObj <- list(  formula = formula,
                     loss = loss,
                     data = data.frame(),
                     n.trees = n.trees,
                     interaction.depth = interaction.depth,
                     shrinkage = shrinkage,
                     bag.fraction = bag.fraction,
                     conjugate.gradient = conjugate.gradient,
                     initF = 0,
                     treematrix = numeric(1),
                     nu = numeric(n.trees),
                     results = numeric(1),
                     ri = numeric(1))
   
   
   mf <- model.frame(formula=formula, data=data)
   x <- model.matrix(attr(mf, "terms"), data=mf)
   y <- model.response(mf)
   ri <- x[1,]
   ri[] <- 0
   pf <- process.features(x)
    
   data.size <- dim(x)[1]
   allowed.rows <- numeric(data.size)
   
   #if classification
   if(gbt.defaultObj$loss == 'binomialDeviance') {
      t<-table(y)
      if(length(t) != 2) {
         stop("The response must contain exactly 2 classes")
      }
      if(as.integer(names(t)[1]) != 0 || as.integer(names(t)[2]) != 1) {
         stop("The response must be either 0 or 1")
      }
      loss.integer <- 1
   }
   else {
      loss.integer <- 0
   }
   
   training.sets.indexes <- vector("list",cv.folds)
   indexes <- 1:data.size
   #if cross-validation
   if(cv.folds > 1) {
      #this cross-validation folds-creating code is adapted from the gbm package source code
      #if classification, create folds that preserve the proportion of positives and negatives examples
      if(gbt.defaultObj$loss == 'binomialDeviance') {
         uc <- names(t)
         if ( min( t ) < cv.folds ){
            stop( paste("The smallest class has only", min(t), "objects in the training set. Can't do", cv.folds, "fold cross-validation."))
         }
         cv.group <- vector( length = data.size )
         for ( i in 1:length( uc ) ){
            cv.group[ y == uc[i] ] <- sample( rep( 1:cv.folds , length = t[i] ) )
         }
      }
      else {
         cv.group <- sample(rep(1:cv.folds, length=data.size))
      }
      valid.size <- table(cv.group)
      #print(valid.size)
      for(f in 1:cv.folds) {
         training.sets.indexes[[f]] <- indexes[(cv.group!=f)]
      }
   }
   else { #no cross validation
      cv.folds <- 1
      valid.size <- c(0)
      training.sets.indexes[[1]] <- 1:data.size
   }
   
   if(store.results || verbose) {
      n.results.per.cv <- floor(n.trees/100)+min(n.trees, 10)
      gbt.defaultObj$results <- numeric(n.results.per.cv * 5)
   }
   if(store.results) {
      results <- numeric(cv.folds * n.results.per.cv * 5)
   }
   
   shrinkage.integer <- shrinkage
   shrinkage.integer$type <- as.integer(shrinkage$type)
   if(shrinkage$type == "fixed") {
      shrinkage.integer$type <- 0
   }
   else if(shrinkage$type == "arithmetic") {
      shrinkage.integer$type <- 1
   }
   else if(shrinkage$type == "geometric") {
      shrinkage.integer$type <- 2
   }
   else if(shrinkage$type == "negative.exp") {
      shrinkage.integer$type <- 3
   }
   else {
      stop("Unkown shrinkage type")
   }
   shrinkage.integer$type <- as.integer(shrinkage.integer$type)
   
   treevector.size <- 6*(2^interaction.depth)
   
   #per training set
   for(f in 1:cv.folds) {
      gbt.obj <- gbt.defaultObj
      
      #current training set size
      training.size <- data.size - valid.size[f]
      #subsample size
      sample.size <- floor(gbt.obj$bag.fraction * training.size)
      
      #start with constant
      if(gbt.defaultObj$loss == 'binomialDeviance') {
         gbt.obj$initF <- log(sum(y[training.sets.indexes[[f]]])/(training.size-sum(y[training.sets.indexes[[f]]])))
      }
      else {
         gbt.obj$initF <- mean(y[training.sets.indexes[[f]]])
      }
      
      if(gbt.obj$n.trees > 0) {
         allowed.rows.training <- allowed.rows
         allowed.rows.valid <- allowed.rows
         training.indexes <- training.sets.indexes[[f]]
         allowed.rows.training[training.indexes] <- 1
         if(valid.size[f] > 0) {
            valid.indexes <- indexes[(cv.group==f)]
            allowed.rows.valid[valid.indexes] <- 1
         }
         gbt.obj$treematrix <- numeric(n.trees*treevector.size)
         
         .Call("gbt", as.integer(loss.integer), as.integer(n.trees), as.integer(dim(x)[2]-1), as.integer(training.size), as.integer(valid.size[f]), as.numeric(x), as.numeric(y), as.integer(pf$ordered.indexes), as.integer(training.indexes-1), as.integer(allowed.rows.training), as.integer(allowed.rows.valid), as.character(pf$type), as.numeric(pf$val), as.integer(sample.size), shrinkage.integer, as.numeric(gbt.obj$initF), as.integer(conjugate.gradient), as.integer(interaction.depth), as.integer(treevector.size), as.integer(verbose || store.results), as.numeric(gbt.obj$treematrix), as.numeric(gbt.obj$nu), as.numeric(gbt.obj$results))
         
         if(verbose) {
            cat("CV: ", f, "\n")
            print(matrix(byrow=TRUE, data=gbt.obj$results, nrow=n.results.per.cv, ncol=5, dimnames=list(seq(1,n.results.per.cv,1), c("Iteration", "Train Error/Deviance", "Step", "Test Error/Deviance", "Improve"))))
            flush.console()
         }
         if(store.results) {
            results[(5*(f-1)*n.results.per.cv+1):(5*f*n.results.per.cv)] <- gbt.obj$results
         }
      }
   }
   if(cv.folds > 1) { #train using the entire training set now
      gbt.obj <- gbt.defaultObj
      training.size <- data.size
      valid.size <- 0
      allowed.rows.training <- allowed.rows
      allowed.rows.valid <- allowed.rows
      training.indexes <- 1:training.size
      allowed.rows.training[] <- 1
      gbt.obj$treematrix <- numeric(n.trees*treevector.size)
      
      .Call("gbt", as.integer(loss.integer), as.integer(n.trees), as.integer(dim(x)[2]-1), as.integer(training.size), as.integer(valid.size), as.numeric(x), as.numeric(y), as.integer(pf$ordered.indexes), as.integer(training.indexes-1), as.integer(allowed.rows.training), as.integer(allowed.rows.valid), as.character(pf$type), as.numeric(pf$val), as.integer(sample.size), shrinkage.integer, as.numeric(gbt.obj$initF), as.integer(conjugate.gradient), as.integer(interaction.depth), as.integer(treevector.size), as.integer(verbose || store.results), as.numeric(gbt.obj$treematrix), as.numeric(gbt.obj$nu), as.numeric(gbt.obj$results))
   }
   
   gbt.model <- gbt.obj
   if(store.results) {
      gbt.model$results <- matrix(byrow=TRUE, data=results, nrow=(cv.folds * n.results.per.cv), ncol=5, dimnames=list(seq(1,cv.folds*n.results.per.cv,1), c("Iteration", "Train Error/Deviance", "Step", "Test Error/Deviance", "Improve")))
   }
   gbt.model$ri <- ri
   
   return(gbt.model)
}
