process.features <-
function(data) {
   nb.features <- dim(data)[2]-1
   nb.rows <- dim(data)[1]
   processed.features <- list(ordered.indexes = data, type = rep("R", nb.features), val = numeric(nb.features))
   for(i in 1:nb.features) {
      start.index <- nb.rows*i+1
      #order the indexes based on the features
      processed.features$ordered.indexes[start.index:(start.index+nb.rows-1)] <- (order(data[start.index:(start.index+nb.rows-1)])-1)
      nb.values <- length(unique(data[start.index:(start.index+nb.rows-1)]))
      #mono value, useless feature...
      if(nb.values == 1) {
         processed.features$type[i] <- "M"
      }
      else if(nb.values == 2) { #if binary feature, the split value is the mean of the 2 values
         processed.features$type[i] <- "B"
         processed.features$val[i] <- (min(data[start.index:(start.index+nb.rows-1)])+max(data[start.index:(start.index+nb.rows-1)]))/2
      }
      else {
         processed.features$type[i] <- "R" #real
      }
   }
   processed.features$type <- paste0(processed.features$type, collapse="")
   return(processed.features)
}
