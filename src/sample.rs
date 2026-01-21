use crate::GaussianKDE;
use ndarray::prelude::*;
use num_traits::{Float, FloatConst, FromPrimitive};
use rand::prelude::*;
use rand_distr::{
    StandardNormal, Uniform,
    uniform::SampleUniform,
    weighted::{Weight, WeightedIndex},
};
use rand_pcg::Pcg64Mcg;

impl<F> GaussianKDE<F>
where
    F: Float + FloatConst + FromPrimitive + SampleUniform + Weight + 'static,
    StandardNormal: Distribution<F>,
{
    /// Sample a random point from the probability density estimated by the KDE.
    ///
    /// This function uses an operating system source to seed the RNG.
    pub fn sample(&self) -> Array1<F> {
        let mut rng = Pcg64Mcg::from_os_rng();
        let i = if let Some(ref w) = self.weights {
            let choice = WeightedIndex::new(w.iter()).unwrap();
            choice.sample(&mut rng)
        } else {
            Uniform::new(0, self.data.dim().0).unwrap().sample(&mut rng)
        };
        let tmp = Array1::from_shape_simple_fn(self.data.dim().1, || rng.sample(StandardNormal));
        return &self.data.index_axis(Axis(0), i) + &self.cholesky.dot(&tmp);
    }

    /// Sample a random point from the probability density estimated by the KDE using a given RNG.
    pub fn sample_with_rng(&self, rng: &mut impl Rng) -> Array1<F> {
        let i = if let Some(ref w) = self.weights {
            let choice = WeightedIndex::new(w.iter()).unwrap();
            choice.sample(rng)
        } else {
            Uniform::new(0, self.data.dim().0).unwrap().sample(rng)
        };
        let tmp = Array1::from_shape_simple_fn(self.data.dim().1, || rng.sample(StandardNormal));
        return &self.data.index_axis(Axis(0), i) + &self.cholesky.dot(&tmp);
    }

    /// Sample `n` random point from the probability density estimated by the KDE.
    ///
    /// This function uses an operating system source to seed the RNG.
    pub fn sample_batch(&self, n: usize) -> Array2<F> {
        let mut rng = Pcg64Mcg::from_os_rng();
        let mut res =
            Array2::from_shape_simple_fn((n, self.data.dim().1), || rng.sample(StandardNormal));
        if let Some(ref w) = self.weights {
            let choice = WeightedIndex::new(w.iter()).unwrap();
            let mut tmp;
            for i in 0..n {
                let k = choice.sample(&mut rng);
                tmp = &self.data.index_axis(Axis(0), k)
                    + &self.cholesky.dot(&res.index_axis(Axis(0), i));
                res.index_axis_mut(Axis(0), i).assign(&tmp);
            }
        } else {
            let uniform = Uniform::new(0, self.data.dim().0).unwrap();
            let mut tmp;
            for i in 0..n {
                let k = uniform.sample(&mut rng);
                tmp = &self.data.index_axis(Axis(0), k)
                    + &self.cholesky.dot(&res.index_axis(Axis(0), i));
                res.index_axis_mut(Axis(0), i).assign(&tmp);
            }
        }
        return res;
    }

    /// Sample `n` random point from the probability density estimated by the KDE using a given RNG.
    pub fn sample_batch_with_rng(&self, n: usize, rng: &mut impl Rng) -> Array2<F> {
        let mut res =
            Array2::from_shape_simple_fn((n, self.data.dim().1), || rng.sample(StandardNormal));
        if let Some(ref w) = self.weights {
            let choice = WeightedIndex::new(w.iter()).unwrap();
            let mut tmp;
            for i in 0..n {
                let k = choice.sample(rng);
                tmp = &self.data.index_axis(Axis(0), k)
                    + &self.cholesky.dot(&res.index_axis(Axis(0), i));
                res.index_axis_mut(Axis(0), i).assign(&tmp);
            }
        } else {
            let uniform = Uniform::new(0, self.data.dim().0).unwrap();
            let mut tmp;
            for i in 0..n {
                let k = uniform.sample(rng);
                tmp = &self.data.index_axis(Axis(0), k)
                    + &self.cholesky.dot(&res.index_axis(Axis(0), i));
                res.index_axis_mut(Axis(0), i).assign(&tmp);
            }
        }
        return res;
    }
}

#[cfg(test)]
mod tests {
    use crate::GaussianKDE;
    use ndarray::prelude::*;

    #[test]
    fn sample_test_1d() {
        let data = array![[0.15], [0.2], [0.21], [0.5], [0.72], [0.74], [0.8]];
        let kde = GaussianKDE::new(data.clone(), None).unwrap();
        let _sample = kde.sample_batch(100_000);
    }

    #[test]
    fn sample_test_2d() {
        let data = array![
            [0.15, 0.4],
            [0.2, 0.3],
            [0.21, 0.29],
            [0.31, 0.74],
            [0.72, 0.9],
            [0.74, 0.84],
            [0.6, 0.3]
        ];
        let kde = GaussianKDE::new(data.clone(), None).unwrap();
        let _sample = kde.sample_batch(100_000);
    }
}
