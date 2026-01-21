use crate::GaussianKDE;
use ndarray::{Zip, prelude::*};
use num_traits::{Float, FloatConst, FromPrimitive};

impl<F> GaussianKDE<F>
where
    F: Float + FloatConst + FromPrimitive + 'static,
{
    /// Evaluate the probability density estimated by the KDE at the point `x`.
    ///
    /// *Panics* if the dimension of `x` does not match the dimension of the KDE dataset.
    pub fn eval(&self, x: ArrayView1<F>) -> F {
        assert_eq!(x.dim(), self.data.dim().1);
        return if let Some(ref w) = self.weights {
            Zip::from(self.data.rows())
                .and(w)
                .fold(F::zero(), |acc, xi, w| {
                    let z: Array1<F> = self.inv_cholesky.dot(&(&xi - &x));
                    acc + *w * F::exp(-F::from(0.5).unwrap() * z.dot(&z))
                })
                * self.normalization
        } else {
            self.data.rows().into_iter().fold(F::zero(), |acc, xi| {
                let z: Array1<F> = self.inv_cholesky.dot(&(&xi - &x));
                acc + F::exp(-F::from(0.5).unwrap() * z.dot(&z))
            }) * self.normalization
        };
    }

    /// Evaluate the probability density estimated by the KDE at multiple points given by the array `x`.
    ///
    /// The points are expected to be given as array of shape `(n_points, dim)`, i.e. a single point is expected to
    /// lie along `Axis(1)`.
    ///
    /// **Panic**s if the dimension of `x` does not match the dimension of the KDE dataset.
    pub fn eval_batch(&self, x: ArrayView2<F>) -> Array1<F> {
        assert_eq!(x.dim().1, self.data.dim().1);
        let mut arg = F::zero();
        let mut tmp = F::zero();
        return if let Some(ref w) = self.weights {
            Array1::from_shape_fn(x.dim().0, |j| {
                Zip::from(self.data.rows())
                    .and(w)
                    .fold(F::zero(), |acc, xi, w| {
                        arg = F::zero();
                        tmp = F::zero();
                        for i in 0..self.inv_cholesky.dim().0 {
                            for k in 0..=i {
                                tmp = self.inv_cholesky[[i, k]] * (xi[[k]] - x[[j, k]]);
                                arg = arg + tmp * tmp;
                            }
                        }
                        acc + *w * F::exp(-F::from(0.5).unwrap() * arg)
                    })
                    * self.normalization
            })
        } else {
            Array1::from_shape_fn(x.dim().0, |j| {
                self.data.rows().into_iter().fold(F::zero(), |acc, xi| {
                    arg = F::zero();
                    tmp = F::zero();
                    for i in 0..self.inv_cholesky.dim().0 {
                        for k in 0..=i {
                            tmp = self.inv_cholesky[[i, k]] * (xi[[k]] - x[[j, k]]);
                            arg = arg + tmp * tmp;
                        }
                    }
                    acc + F::exp(-F::from(0.5).unwrap() * tmp * tmp)
                }) * self.normalization
            })
        };
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::prelude::*;

    use crate::GaussianKDE;

    #[test]
    fn eval_1d_test() {
        let data = array![
            [0.5634880436705391],
            [0.445981611845074],
            [0.7438671296401687]
        ];
        let w_test = array![0.04622972052712859, 0.15162404870472723, 0.291517199926956];
        let kde = GaussianKDE::new(data.clone(), None).unwrap();
        assert_relative_eq!(
            kde.eval(array![0.3].view()),
            0.6318104956149939,
            epsilon = 1E-10
        );
        let kde = GaussianKDE::new(data, Some(w_test)).unwrap();
        assert_relative_eq!(
            kde.eval(array![0.3].view()),
            0.6002942112023868,
            epsilon = 1E-10
        );
    }

    #[test]
    fn eval_3d_test() {
        #[rustfmt::skip]
        let data = array![
            [4.778289487550605452e-01, 6.915810807566095120e-01, 8.973119595652500058e-01],
            [8.092981665695588855e-01, 6.952206389245977336e-01, 8.867610462010474537e-01],
            [4.016505747889576039e-01, 6.735560621931444558e-01, 6.015164821850446097e-01],
            [6.183433169768373094e-01, 9.782506843349931813e-01, 8.643804075625444172e-01],
            [8.470914298329793590e-01, 8.062118291413915561e-01, 7.143558061103683077e-01],
            [4.336121335223386275e-01, 8.069600652351297532e-01, 9.589039393815833590e-01],
            [3.374319617323934262e-01, 5.729598702618347028e-01, 8.259685606489839405e-01],
            [9.510078434543683956e-01, 7.007529367689996347e-01, 1.796766943464989108e-02],
            [2.938782386889049469e-02, 1.078441585862294216e-01, 5.370506790487759030e-01],
            [4.110256667672318454e-02, 2.086942584603000972e-01, 6.946406596087403296e-01]
        ];
        let w_test = array![
            4.545965000176888093e-01,
            6.656013082343981146e-03,
            4.089870211211721340e-01,
            8.170516288204880961e-01,
            7.990649716826044857e-01,
            2.787743020513939740e-01,
            7.606137013085603193e-01,
            6.238816644921463261e-02,
            8.113067447451907110e-01,
            6.972756025050139694e-01,
        ];
        let x_test = array![
            4.184559795606306309e-01,
            1.755027879973122262e-01,
            9.086181878876633533e-01,
        ];
        let kde = GaussianKDE::new(data.clone(), None).unwrap();
        assert_relative_eq!(
            kde.eval(x_test.view()),
            0.012985562962085305,
            epsilon = 1E-10
        );
        let kde = GaussianKDE::new(data, Some(w_test)).unwrap();
        assert_relative_eq!(
            kde.eval(x_test.view()),
            0.00019416613783346587,
            epsilon = 1E-10
        );
    }
}
