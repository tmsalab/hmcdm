# hmcdm 2.1.2

- Release of `hmcdm` package version 2.1.2.

- Resolved compilation warnings on CRAN by replacing the deprecated `arma::conv_to<double>::from()` with `arma::as_scalar()`. This addresses an issue introduced by a recent `RcppArmadillo` update and ensures forward compatibility with the latest version of Armadillo.

# hmcdm 2.1.0

- Release of `hmcdm` package version 2.1.0.

- Allows sparse response arrays and design arrays with missing values.

# hmcdm 2.0.0

- Release of `hmcdm` package version 2.0.0.

# hmcdm 1.2.0.9000

## Deployment

- Switched to GitHub Actions from Travis-CI ([#4](https://github.com/tmsalab/hmcdm/pull/4))

# hmcdm 1.0.0

- Initial release of the `hmcdm` package.
