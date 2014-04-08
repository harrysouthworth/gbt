gbt <-
function(  formula = formula(data),
                  loss = 'squaredLoss',
                  data = list(),
                  n.trees = 100,
                  interaction.depth = 1,
                  shrinkage = list(type='fixed', value = 0.001),
                  bag.fraction = 0.5,
                  cv.folds=0,
                  conjugate.gradient = 0,
                  store.results = 0,
                  verbose = 1)
{
   gbt.obj <- gbt.fit(  formula = formula,
                        loss = loss,
                        data = data,
                        n.trees = n.trees,
                        interaction.depth = interaction.depth,
                        shrinkage = shrinkage,
                        bag.fraction = bag.fraction,
                        cv.folds = cv.folds,
                        conjugate.gradient = conjugate.gradient,
                        store.results = store.results,
                        verbose = verbose)
                        
   class(gbt.obj) <- 'gbt'
   return(gbt.obj)
}
