\name{relative.importance}
\alias{relative.importance}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Methods for estimating relative influence/importance of variables}
\description{
Computes the relative importance of the features.
}
\usage{
relative.importance(object, n.trees)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{
a \code{gbt} object created from an initial call to \code{\link{gbt}}.
}
  \item{n.trees}{
the number of trees to use for computations.
}
}
\value{
Returns a vector of estimated relative influences.
}
\author{
Alexandre Michelis \email{alexandremichelis@gmail.com}
}
