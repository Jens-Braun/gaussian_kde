use criterion::{Criterion, criterion_group, criterion_main};
use gaussian_kde::GaussianKDE;
use ndarray::prelude::*;
use ndarray_npy::read_npy;
use std::path::PathBuf;

const N_GRID: usize = 20;

fn tth_eval_bench(c: &mut Criterion) {
    let pwd = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples");
    let npy: Array2<f64> = read_npy(pwd.join("tth.npy")).unwrap();
    let weights: Array1<f64> = npy.slice(s![.., -1]).to_owned();
    let data: Array2<f64> = npy.slice(s![.., ..-1]).to_owned();
    let kde = GaussianKDE::new(data, Some(weights)).unwrap();
    let kde_margin = kde.marginalize_to(&[2, 3]).unwrap();
    let x = Array1::linspace(0., 1., N_GRID);
    let grid = Array3::from_shape_fn(
        (N_GRID, N_GRID, 2),
        |(i, j, k)| if k == 0 { x[i] } else { x[j] },
    )
    .into_shape_with_order((N_GRID * N_GRID, 2))
    .unwrap();

    c.bench_function("2D eval bench tth 10k points", |b| {
        b.iter(|| kde_margin.eval_batch(grid.view()))
    });
}

fn tth_sample_bench(c: &mut Criterion) {
    let pwd = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples");
    let npy: Array2<f64> = read_npy(pwd.join("tth.npy")).unwrap();
    let weights: Array1<f64> = npy.slice(s![.., -1]).to_owned();
    let data: Array2<f64> = npy.slice(s![.., ..-1]).to_owned();
    let kde = GaussianKDE::new(data, Some(weights)).unwrap();

    c.bench_function("2D sample bench tth 10k points", |b| {
        b.iter(|| kde.sample_batch(100_000))
    });
}

criterion_group!(benches, tth_eval_bench, tth_sample_bench);
criterion_main!(benches);
