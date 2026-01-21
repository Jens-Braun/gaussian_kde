#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, doc(auto_cfg))]
#![allow(clippy::needless_return)]

//! `gaussian_kde` provides multivariate kernel density estimation (KDE) with Gaussian kernels and optionally weighed data points.
//!
//! Given a dataset $X = {x_1, \cdots, x_n}$ sampled from an arbitrary probability density function (PDF), the
//! underlying PDF is estimated as a sum of *kernel* functions $K$ centered at the points of the original dataset:
//! \\[ f_\mathrm{KDE}(x) = \frac{1}{\sum_i w_i} \sum_{i=1}^n w_i \\, K_H\left(\bm{x} - \bm{x}_i\right). \\]
//! Here, $H$ is the *bandwidth* matrix.
//!
//! Specifically, this crate implements KDE with multivariate normal kernels and covariance based bandwidths,
//! \\[ K_H(\bm{y}) = \frac{1}{\sqrt{(2\pi)^d \det H}} \exp\left(- \frac{1}{2} \bm{y}^\top H^{-1} \bm{y}\right) \quad \text{and} \quad H = h^2 V,\\]
//! where $h$ is the scalar bandwidth factor and $V$ is the dataset's covariance matrix. Inserting this into the equation above,
//! the density estimation reads
//! \\[ f_\mathrm{KDE}(x) = \frac{1}{h^d \sqrt{(2\pi)^d \det V} \sum_i w_i} \sum_{i=1}^n w_i \\, \exp\left(- \frac{1}{2h^2}(\bm{x} - \bm{x}_i)^\top V^{-1}(\bm{x} - \bm{x}_i)\right). \\]
//! For more details on (multivariate) kernel density estimation, see e.g. [[1](#ref1), [2](#ref2)].
//!
//! This implementation is largely based on the one in [`scipy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html).
//!
//! ---
//!
//! <a name = "ref1"></a> \[1\] [Gramacki, Artur. Nonparametric Kernel Density Estimation and Its Computational Aspects. Vol. 37. Studies in Big Data. Springer, 2018.](https://doi.org/10.1007/978-3-319-71688-6)
//!
//! <a name = "ref2"></a> \[2\] [Scott, David W. Multivariate Density Estimation: Theory, Practice, and Visualization. Second edition. Wiley, 2014.](https://doi.org/10.1002/9781118575574)

use std::marker::PhantomData;

use ndarray::{Zip, prelude::*};
use num_traits::{Float, FloatConst, FromPrimitive};

pub use bandwidth::{Bandwidth, ScottBandwidth, SilvermanBandwidth};
pub use error::{ErrorKind, KDEError};

use crate::cholesky::{cholesky_decomposition, cholesky_inverse};

mod bandwidth;
mod cholesky;
mod error;
mod eval;
#[cfg(feature = "sample")]
mod sample;

/// Multivariate kernel density estimation with Gaussian kernels and optionally weighed data points.
pub struct GaussianKDE<F, B = bandwidth::ScottBandwidth>
where
    F: Float + FloatConst + FromPrimitive,
    B: Bandwidth<F>,
{
    data: Array2<F>,
    weights: Option<Array1<F>>,
    cholesky: Array2<F>,
    inv_cholesky: Array2<F>,
    normalization: F,
    // The bandwidth is only used as static function during init, but we keep it attached to the struct in order to
    // properly forward it in case of e.g. marginalization.
    _bandwidth_marker: PhantomData<B>,
}

impl<F> GaussianKDE<F>
where
    F: Float + FloatConst + FromPrimitive + 'static,
{
    /// Create a new kernel density estimator from the given dataset and (optionally) weights using the default
    /// bandwidth choice [`ScottBandwidth`]. If no weights are given, all points are weighed equally.
    ///
    /// The dataset is expected to be given as array of shape `(n_points, dim)`, i.e. a single point is expected to
    /// lie along `Axis(1)`.
    pub fn new(
        data: Array2<F>,
        weights: Option<Array1<F>>,
    ) -> Result<GaussianKDE<F, bandwidth::ScottBandwidth>, KDEError> {
        return Self::with_bandwidth(data, weights);
    }

    /// Get a view of the KDE's data.
    pub fn data<'kde>(&'kde self) -> ArrayView2<'kde, F> {
        return self.data.view();
    }

    /// Get a view of the KDE's weights.
    pub fn weights<'kde>(&'kde self) -> Option<ArrayView1<'kde, F>> {
        return self.weights.as_ref().map(|w| w.view());
    }

    /// Get a view of the lower-triangular matrix $L$ obtained from the Cholesky decomposition $V = LL^\top$ of the dataset's
    /// covariance matrix $V$.
    pub fn cholesky<'kde>(&'kde self) -> ArrayView2<'kde, F> {
        return self.cholesky.view();
    }
}

impl<F, B> GaussianKDE<F, B>
where
    F: Float + FloatConst + FromPrimitive + 'static,
    B: Bandwidth<F>,
{
    /// Create a new kernel density estimator from the given dataset and (optionally) weights using the specified
    /// bandwidth factor choice. If no weights are given, all points are weighed equally.
    ///
    /// The dataset is expected to be given as array of shape `(n_points, dim)`, i.e. a single point is expected to
    /// lie along `Axis(1)`.
    pub fn with_bandwidth(
        data: Array2<F>,
        weights: Option<Array1<F>>,
    ) -> Result<GaussianKDE<F, B>, KDEError> {
        let n_samples = data.dim().0;
        let dim = data.dim().1;
        // Preliminary shape checks
        if let Some(ref w) = weights
            && data.dim().0 != w.dim()
        {
            return Err(KDEError::new(
                ErrorKind::ShapeError,
                format!(
                    "expected {} weights for data array with shape `{:?}`, found {}",
                    n_samples,
                    data.dim(),
                    w.dim()
                ),
            ));
        }
        if data.dim().0 < data.dim().1 {
            return Err(KDEError::new(
                ErrorKind::SingularityError,
                format!(
                    "the dataset has fewer entries ({}) than dimensions ({}), resulting in a singular covariance matrix",
                    data.dim().0,
                    data.dim().1
                ),
            ));
        }
        // Prepare values which are repeatedly used during evaluation / sampling
        let sum_weights = if let Some(ref w) = weights {
            w.sum()
        } else {
            F::from(n_samples).unwrap()
        };
        let bw = B::bandwidth(data.view(), weights.as_ref().map(|w| w.view()));
        let cov;
        if let Some(ref w) = weights {
            // Weighted data -> weighted mean / covariance
            let means = Array1::from_shape_fn(dim, |i| {
                Zip::from(data.index_axis(Axis(1), i))
                    .and(w)
                    .fold(F::zero(), |acc, x, w| acc + *w * *x)
                    / sum_weights
            });
            cov = Array2::from_shape_fn((dim, dim), |(i, j)| {
                Zip::from(data.index_axis(Axis(1), i))
                    .and(data.index_axis(Axis(1), j))
                    .and(w)
                    .fold(F::zero(), |acc, x, y, w| {
                        acc + *w * (*x - means[i]) * (*y - means[j])
                    })
                    / (sum_weights
                        - w.iter().map(|w| *w * *w).fold(F::zero(), |acc, x| acc + x) / sum_weights)
                    * bw
                    * bw
            });
        } else {
            let means = Array1::from_shape_fn(dim, |i| data.index_axis(Axis(1), i).mean().unwrap());
            cov = Array2::from_shape_fn((dim, dim), |(i, j)| {
                Zip::from(data.index_axis(Axis(1), i))
                    .and(data.index_axis(Axis(1), j))
                    .fold(F::zero(), |acc, x, y| {
                        acc + (*x - means[i]) * (*y - means[j])
                    })
                    / (sum_weights - F::one())
                    * bw
                    * bw
            });
        }

        let cholesky = cholesky_decomposition(cov.view())?;
        let inv_cholesky = cholesky_inverse(cholesky.view());
        let det = cholesky.diag().product();
        let normalization = F::recip(
            sum_weights * det * F::powi(F::sqrt(F::from(2).unwrap() * F::PI()), dim as i32),
        );
        return Ok(Self {
            data,
            weights,
            cholesky,
            inv_cholesky,
            normalization,
            _bandwidth_marker: PhantomData,
        });
    }

    /// Marginalize the density by integrating out the components given in `dims`. For Gaussian kernels, this is
    /// equivalent to simply remove the marginalized components from the dataset.
    pub fn marginalize(&self, dims: &[usize]) -> Result<Self, KDEError> {
        for i in dims {
            if *i > self.data.dim().0 {
                return Err(KDEError::new(
                    ErrorKind::IndexError,
                    format!(
                        "index `{i}` out of bounds for data of dimension `{}`",
                        self.data.dim().0
                    ),
                ));
            }
        }
        let indices = (0..self.data.dim().0)
            .filter(|i| dims.contains(i))
            .collect::<Vec<_>>();
        let marginalized = self.data.select(Axis(1), &indices);
        return Ok(Self::with_bandwidth(marginalized, self.weights.clone()).unwrap());
    }

    /// Marginalize the density by integrating out all components but the ones given in `dims`. For Gaussian kernels,
    /// this is equivalent to simply remove the marginalized components from the dataset.
    pub fn marginalize_to(&self, dims: &[usize]) -> Result<Self, KDEError> {
        for i in dims {
            if *i > self.data.dim().0 {
                return Err(KDEError::new(
                    ErrorKind::IndexError,
                    format!(
                        "index `{i}` out of bounds for data of dimension `{}`",
                        self.data.dim().0
                    ),
                ));
            }
        }
        let marginalized = self.data.select(Axis(1), dims);
        return Ok(Self::with_bandwidth(marginalized, self.weights.clone()).unwrap());
    }
}
