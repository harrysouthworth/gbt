\name{gbt.fit}
\alias{gbt.fit}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Fits a gbt model. Not intended for end-user. Shouldn't be used. Use \code{\link{gbt}} instead.
}
\description{
Fits gradient boosted trees.
}
\usage{
gbt.fit(formula, loss = "squaredLoss", data, n.trees, interaction.depth, shrinkage, bag.fraction, cv.folds, conjugate.gradient, store.results, verbose)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{formula}{
a symbolic description of the model to be fit
}
  \item{loss}{
a character string specifying the name of the loss function to use. Currently available options are "squaredLoss" for regression, and "binomialDeviance" for binary classification.
}
  \item{data}{
a data frame containing the variables in the model. By
   default the variables are taken from \code{environment(formula)}, typically 
   the environment from which \code{gbt} is called. Examples with missing values are ignored.
}
  \item{n.trees}{
the total number of trees to fit. This is equivalent to the
number of iterations and the number of trees in the additive
expansion.
}
  \item{interaction.depth}{
The maximum depth of variable interactions. 1 implies
an additive model, 2 implies a model with up to 2-way interactions, etc.
}
  \item{shrinkage}{
a shrinkage parameter way of evolving applied to the trees in the expansion.
Also known as the learning rate or step-size reduction.
Currently available options are:
list(type="fixed", value=v) where v is the fixed learning rate
list(type="arithmetic", start=v1, end=v2) where v1 is the learning rate of the first iteration and v2 of the last. In between, the learning rate follows an arithmetic decrease (constant step size).
list(type="geometric", start=v1, end=v2) where v1 is the learning rate of the first iteration and v2 of the last. In between, the learning rate follows an geometric decrease.
list(type="negative.exp", start=v1, end=v2, iter75=m) where v1 is the learning rate of the first iteration and v2 of the last. iter75 is the iteration number at which the learning rate will have reached 75% of the decrease. The learning rate follows a negative exponential decrease.
}
  \item{bag.fraction}{
the fraction of the training set observations randomly
selected to propose the next tree in the expansion. This introduces randomnesses
into the model fit. If \code{bag.fraction}<1 then running the same model twice
will result in similar but different fits. \code{gbt} uses the R random number
generator so \code{set.seed} can ensure that the model can be
reconstructed.
}
  \item{cv.folds}{
Number of cross-validation folds to perform. If \code{cv.folds}>1 then
\code{gbm}, in addition to the usual fit, will perform a cross-validation.
}
  \item{conjugate.gradient}{
Will perform a conjugate gradient descent using the Polak-Ribiere rule instead of the classic gradient descent. This functionality is experimental.
}
  \item{store.results}{
If 1, the results (train (and test, in case of cross-validation) deviances, and step sizes, for each cross-validation) as displayed when \code{verbose} is activated are stored in \code{gbt$results}. They can be used to produce an estimate of the generalization error in order to select the optimal number of iterations.
}
  \item{verbose}{
If 1, will display results about each model that is built, once it is built.
}
}
\details{
Not intended for end-users. Shouldn't be used. Use \code{\link{gbt}} instead.
}
\value{
Returns an object with no particular class.
}
\author{
Alexandre Michelis <alexandremichelis@gmail.com>
}
