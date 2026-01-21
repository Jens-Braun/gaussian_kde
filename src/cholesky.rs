//! Cholesky decomposition and matrix inversion are in principle already provided for `ndarray` via the
//! `ndarray-linalg` crate. However, this crate uses LINPACK/BLAS for high-performance implementations of said
//! algorithms, which carries several dependencies. To avoid these dependencies for the two simple algorithms
//! required here, they are reimplemented here.

use ndarray::prelude::*;
use num_traits::{Float, FloatConst, FromPrimitive};

use crate::KDEError;

pub(crate) fn cholesky_decomposition<F>(m: ArrayView2<F>) -> Result<Array2<F>, KDEError>
where
    F: Float + FloatConst + FromPrimitive + 'static,
{
    let mut res = Array2::zeros(m.raw_dim());
    let mut tmp;
    for i in 0..m.dim().0 {
        for j in 0..i {
            res[[i, j]] =
                (m[[i, j]] - res.slice(s![i, ..=i]).dot(&res.slice(s![j, ..=i]))) / res[[j, j]];
        }
        tmp = m[[i, i]] - res.slice(s![i, ..i]).dot(&res.slice(s![i, ..i]));
        if tmp <= F::zero() {
            return Err(KDEError::new(
                crate::ErrorKind::SingularityError,
                "the covariance matrix appears to not be positive-definite",
            ));
        }
        res[[i, i]] = F::sqrt(tmp);
    }
    return Ok(res);
}

pub(crate) fn cholesky_inverse<F>(m: ArrayView2<F>) -> Array2<F>
where
    F: Float + FloatConst + FromPrimitive + 'static,
{
    let mut res = Array2::zeros(m.raw_dim());
    for j in 0..m.dim().0 {
        res[[j, j]] = m[[j, j]].recip();
        for i in (j + 1)..m.dim().0 {
            res[[i, j]] = -m.slice(s![i, ..i]).dot(&res.slice(s![..i, j])) / m[[i, i]];
        }
    }
    return res;
}

#[cfg(test)]
mod tests {
    use crate::cholesky::cholesky_inverse;

    use super::cholesky_decomposition;
    use approx::assert_relative_eq;
    use ndarray::prelude::*;

    #[test]
    fn cholesky_test() {
        #[rustfmt::skip]
        let m = array![
            [2.0542973311512913, 1.9711183024062744, 1.2585589835749347, 1.5690403985438703, 1.5375107006597937],
            [1.9711183024062744, 2.1801084607977526, 1.2007754146681564, 1.543014147477119 , 1.3000841475915699],
            [1.2585589835749347, 1.2007754146681564, 1.1775143453668928, 1.0301046678804817, 0.8521784444215207],
            [1.5690403985438703, 1.543014147477119 , 1.0301046678804817, 1.7990191449143589, 1.2738543283689894],
            [1.5375107006597937, 1.3000841475915699, 0.8521784444215207, 1.2738543283689894, 1.4017934591929233]
        ];
        #[rustfmt::skip]
        let res_numpy = array![
            [ 1.4332820138239688  ,  0.                  ,  0.                  ,  0.                  ,  0.                  ],
            [ 1.375248055438419   ,  0.5374023137376688  ,  0.                  ,  0.                  ,  0.                  ],
            [ 0.8780958467602085  , -0.012698477291453415,  0.6374172714508477  ,  0.                  ,  0.                  ],
            [ 1.0947185434621487  ,  0.06978868227710758 ,  0.10938371866419833 ,  0.7640518290218844  ,  0.                  ],
            [ 1.0727202921899122  , -0.3259612841143241  , -0.1473320618318058  ,  0.18112864153606167 ,  0.30049884183993686 ]
        ];
        let res = cholesky_decomposition(m.view()).unwrap();
        assert_relative_eq!(res_numpy.as_slice().unwrap(), res.as_slice().unwrap());
    }

    #[test]
    fn inverse_test() {
        #[rustfmt::skip]
        let m = array![
            [ 1.4332820138239688  ,  0.                  ,  0.                  ,  0.                  ,  0.                  ],
            [ 1.375248055438419   ,  0.5374023137376688  ,  0.                  ,  0.                  ,  0.                  ],
            [ 0.8780958467602085  , -0.012698477291453415,  0.6374172714508477  ,  0.                  ,  0.                  ],
            [ 1.0947185434621487  ,  0.06978868227710758 ,  0.10938371866419833 ,  0.7640518290218844  ,  0.                  ],
            [ 1.0727202921899122  , -0.3259612841143241  , -0.1473320618318058  ,  0.18112864153606167 ,  0.30049884183993686 ]
        ];
        #[rustfmt::skip]
        let inv_numpy = array![
            [ 0.6976993992494326 ,  0.                 ,  0.                 ,  0.                 ,  0.                 ],
            [-1.7854588965664873 ,  1.860803302175857  ,  0.                 ,  0.                 ,  0.                 ],
            [-0.9967090357856467 ,  0.03707048668881848,  1.5688310386128463 ,  0.                 ,  0.                 ],
            [-0.6938742707908822 , -0.1752733427703402 , -0.22459807887505817,  1.3088117350365736 ,  0.                 ],
            [-4.497832547471792  ,  2.1422995297757876 ,  0.9045634080932657 , -0.7888991855745792 ,  3.3277998473373755 ]
        ];
        let res = cholesky_inverse(m.view());
        assert_relative_eq!(inv_numpy.as_slice().unwrap(), res.as_slice().unwrap());
    }
}
