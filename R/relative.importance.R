relative.importance <-
function(object, n.trees) {
   ri <- numeric(length(object$ri))
   
   treevector.size <- 6*(2^object$interaction.depth)
   
   .Call("ri", as.numeric(object$treematrix), as.numeric(ri), as.integer(treevector.size), as.integer(n.trees), as.integer(object$interaction.depth))
   ri <- sqrt(ri/n.trees)
   ri <- ri / max(ri) * 100
   names(ri) <- names(object$ri)
   return(ri)
}
