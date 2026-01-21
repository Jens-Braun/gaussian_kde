use ndarray::prelude::*;
use num_traits::{Float, FloatConst, FromPrimitive};

/// General trait to customize the selection of the scalar bandwidth $h$.
pub trait Bandwidth<F>
where
    F: Float + FloatConst + FromPrimitive,
{
    fn bandwidth(data: ArrayView2<F>, weights: Option<ArrayView1<F>>) -> F;
}

/// Select the scalar bandwidth factor according to Scott's rule.
///
/// Scott's rule calculates the scalar bandwidth factor according to
/// \\[ h = n_\mathrm{eff}^{-\frac{1}{d+4}}, \\]
/// where $d$ is the dimension of the dataset and
/// \\[ n_\mathrm{eff} = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2} \\]
/// is the effective number of entries in the (weighted) dataset.
pub struct ScottBandwidth {}

impl<F> Bandwidth<F> for ScottBandwidth
where
    F: Float + FloatConst + FromPrimitive,
{
    fn bandwidth(data: ArrayView2<F>, weights: Option<ArrayView1<F>>) -> F {
        let n_samples = data.dim().0;
        let d = data.dim().1;
        let n_eff = if let Some(ref w) = weights {
            F::powi(w.sum(), 2) / w.fold(F::zero(), |acc, w| acc + *w * *w)
        } else {
            F::from(n_samples).unwrap()
        };
        return F::powf(n_eff, -F::from(d + 4).unwrap().recip());
    }
}

/// Select the scalar bandwidth factor according to Silverman's rule of thumb.
///
/// Silverman's rule of thumb calculates the scalar bandwidth factor according to
/// \\[ h = \left(\frac{d+2}{4} \\; n_\mathrm{eff}\right)^{-\frac{1}{d+4}}, \\]
/// where $d$ is the dimension of the dataset and
/// \\[ n_\mathrm{eff} = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2} \\]
/// is the effective number of entries in the (weighted) dataset.
pub struct SilvermanBandwidth {}

impl<F> Bandwidth<F> for SilvermanBandwidth
where
    F: Float + FloatConst + FromPrimitive,
{
    fn bandwidth(data: ArrayView2<F>, weights: Option<ArrayView1<F>>) -> F {
        let n_samples = data.dim().0;
        let d = data.dim().1;
        let n_eff = if let Some(ref w) = weights {
            F::powi(w.sum(), 2) / w.fold(F::zero(), |acc, w| acc + *w * *w)
        } else {
            F::from(n_samples).unwrap()
        };
        return F::powf(
            F::from(0.25).unwrap() * n_eff * F::from(d + 2).unwrap(),
            -F::from(d + 4).unwrap().recip(),
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Bandwidth,
        bandwidth::{ScottBandwidth, SilvermanBandwidth},
    };
    use approx::assert_relative_eq;
    use ndarray::prelude::*;

    #[test]
    fn scott_1d_test() {
        let data = array![
            [0.5634880436705391],
            [0.445981611845074],
            [0.7438671296401687]
        ];
        let w_test = array![0.04622972052712859, 0.15162404870472723, 0.291517199926956];
        assert_relative_eq!(
            ScottBandwidth::bandwidth(data.view(), None),
            0.8027415617602307,
            epsilon = 1E-10
        );
        assert_relative_eq!(
            ScottBandwidth::bandwidth(data.view(), Some(w_test.view())),
            0.8560705025393376,
            epsilon = 1E-10
        );
    }

    #[test]
    fn scott_3d_test() {
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
        assert_relative_eq!(
            ScottBandwidth::bandwidth(data.view(), None),
            0.719685673001152,
            epsilon = 1E-10
        );
        assert_relative_eq!(
            ScottBandwidth::bandwidth(data.view(), Some(w_test.view())),
            0.7505290905335276,
            epsilon = 1E-10
        );
    }

    #[test]
    fn silverman_1d_test() {
        let data = array![
            [0.5634880436705391],
            [0.445981611845074],
            [0.7438671296401687]
        ];
        let w_test = array![0.04622972052712859, 0.15162404870472723, 0.291517199926956];
        assert_relative_eq!(
            SilvermanBandwidth::bandwidth(data.view(), None),
            0.8502830004171938,
            epsilon = 1E-10
        );
        assert_relative_eq!(
            SilvermanBandwidth::bandwidth(data.view(), Some(w_test.view())),
            0.9067702859083041,
            epsilon = 1E-10
        );
    }

    #[test]
    fn silverman_3d_test() {
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
        assert_relative_eq!(
            SilvermanBandwidth::bandwidth(data.view(), None),
            0.6971055968511698,
            epsilon = 1E-10
        );
        assert_relative_eq!(
            SilvermanBandwidth::bandwidth(data.view(), Some(w_test.view())),
            0.7269813048087493,
            epsilon = 1E-10
        );
    }
}
