\name{predict.gbt}
\alias{predict.gbt}
\title{ Predict method for GBT models}
\description{
  Predicted values based on a gradient boosted trees object
}
\usage{
\method{predict}{gbt}(object,
        newdata,
        n.trees,
        ...)
}
\arguments{
  \item{object}{ Object of class inheriting from (\code{\link{gbt}}) }
  \item{newdata}{ Data frame of observations for which to make predictions }
  \item{n.trees}{ Number of trees used in the prediction.}
  \item{\dots}{ further arguments passed to or from other methods }
}
\details{
\code{predict.gbt} produces predicted values for each observation in \code{newdata} using the the first \code{n.trees} iterations of the boosting sequence.
The user has the responsibility to make
sure that \code{newdata} is of the same format (order and number of variables)
as the one originally used to fit the model.
}
\value{

Returns a vector of predictions. By default the predictions are on the scale of f(x). For example, for the binomial deviance loss the returned value is on the log odds scale.
}
\author{
Alexandre Michelis <alexandremichelis@gmail.com>
}
