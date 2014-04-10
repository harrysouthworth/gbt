predict.gbt <-
function(object, newdata, n.trees,...) {
   predictions <- numeric(length(newdata[,1]))
   mf <- model.frame(formula=object$formula, data=newdata)
   x <- model.matrix(attr(mf, "terms"), data=mf)
   treevector.size <- 6*(2^object$interaction.depth)
   .Call("predict", as.numeric(object$treematrix), as.numeric(object$nu), as.numeric(x), as.integer(length(x[,1])), as.integer(n.trees), as.integer(treevector.size), as.integer(object$interaction.depth), as.numeric(predictions), as.numeric(object$initF))
   return(predictions)
}
