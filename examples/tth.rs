//! The dataset used in this example is five-dimensional and has a different weight for each point.
//!
//! It is derived from the high-energy scattering amplitude of the process gg -> ttH at leading order. See
//! https://arxiv.org/abs/2601.00950 for details.
//!
//! When marginalizing the KDE derived from this dataset to the $x_3$-$x_4$-plane, it features a characteristic
//! "X"-shape, which can easily be checked in a contour plot. This example generates a `numpy` file containing
//! the PDF evaluated on a regular grid, the respective contour plot can then be generated with `tth.py`.

use gaussian_kde::GaussianKDE;
use ndarray::prelude::*;
use ndarray_npy::{read_npy, write_npy};
use std::path::PathBuf;

const N_GRID: usize = 50;

fn main() {
    let pwd = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples");
    let npy: Array2<f64> = read_npy(pwd.join("tth.npy")).unwrap();
    let weights: Array1<f64> = npy.slice(s![.., -1]).to_owned();
    let data: Array2<f64> = npy.slice(s![.., ..-1]).to_owned();

    let kde = GaussianKDE::new(data, Some(weights)).unwrap();

    // Marginalize the PDF to the $x_3$-$x_4$-plane and evaluate it on a 2D grid for a contour plot
    let kde_margin = kde.marginalize_to(&[2, 3]).unwrap();
    let x = Array1::linspace(0., 1., N_GRID);
    let mut grid = Array3::from_shape_fn(
        (N_GRID, N_GRID, 2),
        |(i, j, k)| if k == 0 { x[i] } else { x[j] },
    )
    .into_shape_with_order((N_GRID * N_GRID, 2))
    .unwrap();
    let res = kde_margin.eval_batch(grid.view());
    grid.push_column(res.view()).unwrap();
    write_npy(pwd.join("pdf_grid.npy"), &grid).unwrap();
}
