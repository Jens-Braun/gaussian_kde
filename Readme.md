# gaussian_kde
`gaussian_kde` provides multivariate kernel density estimation (KDE) with Gaussian kernels and optionally weighed data points.

Given a dataset $X = {x_1, \cdots, x_n}$ sampled from an arbitrary probability density function (PDF), the
underlying PDF is estimated as a sum of *kernel* functions $K$ centered at the points of the original dataset:
$$ f_\mathrm{KDE}(x) = \frac{1}{\sum_i w_i} \sum_{i=1}^n w_i \, K_H\left(\bm{x} - \bm{x}_i\right). $$
Here, $H$ is the *bandwidth* matrix.

Specifically, this crate implements KDE with multivariate normal kernels and covariance based bandwidths,
$$ K_H(\bm{y}) = \frac{1}{\sqrt{(2\pi)^d \det H}} \exp\left(- \frac{1}{2} \bm{y}^\top H^{-1} \bm{y}\right) \quad \text{and} \quad H = h^2 V,$$
where $h$ is the scalar bandwidth factor and $V$ is the dataset's covariance matrix. Inserting this into the equation above,
the density estimation reads
$$ f_\mathrm{KDE}(x) = \frac{1}{h^d \sqrt{(2\pi)^d \det V} \sum_i w_i} \sum_{i=1}^n w_i \, \exp\left(- \frac{1}{2h^2}(\bm{x} - \bm{x}_i)^\top V^{-1}(\bm{x} - \bm{x}_i)\right). $$
For more details on (multivariate) kernel density estimation, see e.g. [[1](#ref1), [2](#ref2)].

This implementation is largely based on the one in [`scipy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html).

## Usage
Add `gaussian_kde` to a Rust project either with `cargo add gaussian_kde` or by directly adding it to `Cargo.toml`,
```toml
[dependencies]
gaussian_kde = "0.1"
```

### One-dimensional example
The central object of this crate is the `GaussianKDE` struct, which constructs a KDE of a given (weighted) dataset and allows evaluating the estimated PDF or sampling from it.
```rust
use gaussian_kde::GaussianKDE;
use ndarray::prelude::*;

let data = array![[0.563488], [0.445981], [0.743867]];
let weights = array![1., 0.2, 1.5];
let kde = GaussianKDE::new(data, Some(weights)).unwrap();
let pdf = kde.eval(array![0.6].view());
let sample = kde.sample_batch(100);
```

### Five-dimensional example
The [`tth`](examples/tth.rs) example constructs a KDE for a five-dimensional dataset containing 10k points. This KDE is marginalized to a two-dimensional subspace and evaluated on a grid. 

 ---

<a name = "ref1"></a> \[1\] [Gramacki, Artur. Nonparametric Kernel Density Estimation and Its Computational Aspects. Vol. 37. Studies in Big Data. Springer, 2018.](https://doi.org/10.1007/978-3-319-71688-6)

<a name = "ref2"></a> \[2\] [Scott, David W. Multivariate Density Estimation: Theory, Practice, and Visualization. Second edition. Wiley, 2014.](https://doi.org/10.1002/9781118575574)
