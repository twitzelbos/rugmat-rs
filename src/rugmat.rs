use faer::prelude::*;
use rayon::prelude::*;
use rug::Assign;
use rug::Float; // or faer::Mat if needed directly
use rug::ops::{CompleteRound, Pow};
use std::ops::{Index, IndexMut};

pub enum PseudoInverseAlgorithm {
    GradientDescent { alpha: Float },
    LSQR,
    ConjugateGradient,
}

#[macro_export]
macro_rules! t {
    ($mat:expr) => {
        Transpose::new(&$mat)
    };
}
#[derive(Debug)]
pub struct SVD {
    pub u: RugMat,
    pub s: Vec<Float>,
    pub vt: RugMat,
}

pub struct Transpose<'a> {
    pub mat: &'a RugMat,
}

impl<'a> Transpose<'a> {
    pub fn new(mat: &'a RugMat) -> Self {
        Transpose { mat }
    }

    pub fn mul(&self, x: &[Float]) -> Vec<Float> {
        self.mat.matmul_transpose_vec(x)
    }

    pub fn rows(&self) -> usize {
        self.mat.cols
    }

    pub fn cols(&self) -> usize {
        self.mat.rows
    }

    pub fn get(&self, i: usize, j: usize) -> &Float {
        &self.mat[(j, i)]
    }
}

#[derive(Debug, Clone)]
pub struct RugMat {
    pub data: Vec<Float>,
    pub rows: usize,
    pub cols: usize,
}

impl RugMat {
    pub fn new(rows: usize, cols: usize, precision: u32) -> Self {
        let data = vec![Float::with_val(precision, 0); rows * cols];
        Self { data, rows, cols }
    }

    pub fn from_2d_vec(vv: Vec<Vec<Float>>) -> Self {
        let rows = vv.len();
        let cols = vv[0].len();
        let mut data = Vec::with_capacity(rows * cols);
        for j in 0..cols {
            for i in 0..rows {
                data.push(vv[i][j].clone());
            }
        }
        Self { data, rows, cols }
    }

    pub fn get(&self, row: usize, col: usize) -> &Float {
        &self.data[col * self.rows + row]
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut Float {
        &mut self.data[col * self.rows + row]
    }

    /*
    pub fn get_column(&self, j: usize) -> Vec<Float> {
        (0..self.rows).map(|i| self[(i, j)].clone()).collect()
    }

    pub fn get_row(&self, i: usize) -> Vec<Float> {
        (0..self.cols).map(|j| self[(i, j)].clone()).collect()
    }

    pub fn from_columns(cols: &[Vec<Float>]) -> Self {
        let cols_len = cols.len();
        let rows = cols[0].len();
        let mut data = Vec::with_capacity(rows * cols_len);
        for j in 0..cols_len {
            for i in 0..rows {
                data.push(cols[j][i].clone());
            }
        }
        Self {
            data,
            rows,
            cols: cols_len,
        }
    }

    pub fn from_rows(rows_vec: &[Vec<Float>]) -> Self {
        let rows = rows_vec.len();
        let cols = rows_vec[0].len();
        let mut data = Vec::with_capacity(rows * cols);
        for j in 0..cols {
            for i in 0..rows {
                data.push(rows_vec[i][j].clone());
            }
        }
        Self { data, rows, cols }
    }
    */
}

impl Index<(usize, usize)> for RugMat {
    type Output = Float;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (i, j) = index;
        &self.data[j * self.rows + i]
    }
}

impl IndexMut<(usize, usize)> for RugMat {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (i, j) = index;
        &mut self.data[j * self.rows + i]
    }
}

pub fn dot(a: &[Float], b: &[Float]) -> Float {
    assert_eq!(a.len(), b.len());
    let precision = a[0].prec().max(b[0].prec());
    let mut acc = Float::with_val(precision, 0);
    for (x, y) in a.iter().zip(b.iter()) {
        acc += x * y;
    }
    acc
}

pub fn norm2(a: &[Float]) -> Float {
    let precision = a[0].prec();
    let mut acc = Float::with_val(precision, 0);
    for x in a {
        acc += x * x;
    }
    acc
}

/// Reuse buffer: subtract `b` from `a` and store in `out`
pub fn vec_sub_in_place(a: &[Float], b: &[Float], out: &mut [Float]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());

    for i in 0..a.len() {
        out[i].assign(&a[i]);
        out[i] -= &b[i];
    }
}

impl RugMat {
    pub fn matmul(&self, other: &RugMat) -> RugMat {
        assert_eq!(self.cols, other.rows);
        let precision = self.data[0].prec();
        let m = self.rows;
        let k = self.cols;
        let n = other.cols;

        let block_size = 32;
        let mut result = RugMat::new(m, n, precision);
        let mut result_data = vec![Float::with_val(precision, 0); m * n];

        result_data
            .par_chunks_mut(m)
            .enumerate()
            .for_each(|(j, col)| {
                let mut acc = vec![Float::with_val(precision, 0); m];
                for l_block in (0..k).step_by(block_size) {
                    let l_max = (l_block + block_size).min(k);
                    for l in l_block..l_max {
                        for i in 0..m {
                            let a = &self[(i, l)];
                            let b = &other[(l, j)];
                            acc[i] += a * b;
                        }
                    }
                }
                for i in 0..m {
                    col[i] = acc[i].clone();
                }
            });

        result.data = result_data;
        result
    }

    pub fn matmul_vec(&self, v: &[Float]) -> Vec<Float> {
        assert_eq!(self.cols, v.len());
        let precision = self.data[0].prec();
        let m = self.rows;

        (0..m)
            .into_par_iter()
            .map(|i| {
                let mut sum = Float::with_val(precision, 0);
                for j in 0..self.cols {
                    sum += &self[(i, j)] * &v[j];
                }
                sum
            })
            .collect()
    }

    pub fn matmul_transpose_vec(&self, v: &[Float]) -> Vec<Float> {
        assert_eq!(self.rows, v.len());
        let precision = self.data[0].prec();
        let n = self.cols;

        (0..n)
            .into_par_iter()
            .map(|j| {
                let mut sum = Float::with_val(precision, 0);
                for i in 0..self.rows {
                    sum += &self[(i, j)] * &v[i];
                }
                sum
            })
            .collect()
    }

    pub fn dot_columns(&self, i: usize, j: usize) -> Float {
        let precision = self[(0, i)].prec().max(self[(0, j)].prec());
        let mut acc = Float::with_val(precision, 0);
        for row in 0..self.rows {
            acc += &self[(row, i)] * &self[(row, j)];
        }
        acc
    }

    pub fn column_norm2(&self, j: usize) -> Float {
        let precision = self[(0, j)].prec();
        let mut acc = Float::with_val(precision, 0);
        for row in 0..self.rows {
            let x = &self[(row, j)];
            acc += x * x;
        }
        acc
    }

    /// Infinity norm (max absolute row sum), optimized for column-major layout with Rayon
    pub fn norm_inf(&self) -> Float {
        let precision = self.data[0].prec();
        let rows = self.rows;
        let cols = self.cols;

        // Parallel over columns, accumulate per-row values
        let row_sums = (0..cols)
            .into_par_iter()
            .map_init(
                || vec![Float::with_val(precision, 0); rows],
                |local_row_sums, j| {
                    let mut tmp = Float::with_val(precision, 0);
                    for i in 0..rows {
                        tmp.assign(&self[(i, j)]);
                        tmp.abs_mut();
                        local_row_sums[i] += &tmp;
                    }
                    local_row_sums.clone()
                },
            )
            .reduce(
                || vec![Float::with_val(precision, 0); rows],
                |mut acc, other| {
                    for i in 0..rows {
                        acc[i] += &other[i];
                    }
                    acc
                },
            );

        row_sums
            .into_iter()
            .reduce(|a, b| if a > b { a } else { b })
            .unwrap()
    }
}

use crate::float_serializer::{read_float, write_float};
use rayon::prelude::*;
use rug::float::Round;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

impl RugMat {
    /// Compute 2-norm of a vector
    pub fn norm2_vec(v: &[Float]) -> Float {
        let precision = v[0].prec();
        let mut acc = Float::with_val(precision * 2, 0);
        for x in v {
            acc += x * x;
        }
        acc.sqrt()
    }

    /// Solve A x = b approximately using the specified algorithm
    pub fn pseudo_inverse_solve(
        &self,
        b: &[Float],
        iters: usize,
        alg: PseudoInverseAlgorithm,
    ) -> Vec<Float> {
        match alg {
            PseudoInverseAlgorithm::GradientDescent { alpha } => {
                let precision = b[0].prec();
                let mut x = vec![Float::with_val(precision, 0); self.cols];

                for _ in 0..iters {
                    let Ax = self.matmul_vec(&x);
                    let mut r = vec![Float::with_val(precision, 0); self.rows];
                    for i in 0..self.rows {
                        r[i] = Ax[i].clone() - &b[i];
                    }
                    let At_r = t!(self).mul(&r);
                    for j in 0..self.cols {
                        x[j] -= &alpha * &At_r[j];
                    }
                }
                x
            }
            PseudoInverseAlgorithm::LSQR => self.lsqr(b, iters),
            PseudoInverseAlgorithm::ConjugateGradient => self.conjugate_gradient(b, iters),
        }
    }

    pub fn conjugate_gradient(&self, b: &[Float], max_iters: usize) -> Vec<Float> {
        let precision = b[0].prec();

        // Normal equations: AᵗA x = Aᵗb
        let At_b = t!(self).mul(b);
        let mut x = vec![Float::with_val(precision, 0); self.cols];

        let Ax = self.matmul_vec(&x);
        let At_Ax = t!(self).mul(&Ax);

        let mut r: Vec<Float> = At_b
            .iter()
            .zip(&At_Ax)
            .map(|(b, ax)| (b - ax).complete(b.prec()))
            .collect();
        let mut p = r.clone();
        let mut rs_old = r
            .iter()
            .map(|v| (v * v).complete(v.prec()))
            .reduce(|a, b| (a + b))
            .unwrap();

        let initial_norm = rs_old.clone().sqrt();
        let epsilon = Float::with_val(precision, 1e-30);
        let mut restart_regularized = false;

        for iter in 0..max_iters {
            let Ap = {
                let Av = self.matmul_vec(&p);
                t!(self).mul(&Av)
            };
            let denom = p
                .iter()
                .zip(&Ap)
                .map(|(a, b)| (a * b).complete(a.prec()))
                .reduce(|a, b| a + b)
                .unwrap();

            if denom == 0 {
                // Likely rank deficiency
                if !restart_regularized {
                    eprintln!(
                        "[CG] Detected possible rank deficiency at iter {} — restarting with Tikhonov regularization",
                        iter
                    );
                    return self.cg_regularized(b, max_iters, Float::with_val(precision, 1e-10));
                } else {
                    break;
                }
            }

            let alpha = (&rs_old / &denom).complete(rs_old.prec());

            for i in 0..x.len() {
                x[i] += &alpha * &p[i];
                r[i] -= &alpha * &Ap[i];
            }

            let rs_new = r
                .iter()
                .map(|v| (v * v).complete(v.prec()))
                .reduce(|a, b| a + b)
                .unwrap();
            if rs_new < (&epsilon * &initial_norm).complete(epsilon.prec()) {
                break;
            }

            let beta = (&rs_new / &rs_old).complete(rs_new.prec());
            for i in 0..p.len() {
                p[i] = (&r[i] + (&beta * &p[i]).complete(beta.prec()));
            }
            rs_old = rs_new;
        }
        x
    }

    /// Regularized CG: Solve (AᵗA + λI)x = Aᵗb
    pub fn cg_regularized(&self, b: &[Float], max_iters: usize, lambda: Float) -> Vec<Float> {
        let precision = b[0].prec();

        let At_b = t!(self).mul(b);
        let mut x = vec![Float::with_val(precision, 0); self.cols];

        let Ax = self.matmul_vec(&x);
        let At_Ax = t!(self).mul(&Ax);

        let mut r: Vec<Float> = At_b
            .iter()
            .zip(&At_Ax)
            .map(|(b, ax)| (b - ax).complete(b.prec()))
            .collect();
        for (ri, xi) in r.iter_mut().zip(&x) {
            *ri -= &lambda * xi;
        }

        let mut p = r.clone();
        let mut rs_old = r
            .iter()
            .map(|v| (v * v).complete(v.prec()))
            .reduce(|a, b| a + b)
            .unwrap();

        for _ in 0..max_iters {
            let Ap = {
                let Av = self.matmul_vec(&p);
                let AtAv = t!(self).mul(&Av);
                AtAv.into_iter()
                    .zip(&p)
                    .map(|(val, pi)| val + &lambda * pi)
                    .collect::<Vec<_>>()
            };
            let denom = p
                .iter()
                .zip(&Ap)
                .map(|(a, b)| (a * b).complete(a.prec()))
                .reduce(|a, b| a + b)
                .unwrap();
            let alpha = (&rs_old / &denom).complete(rs_old.prec());

            for i in 0..x.len() {
                x[i] += &alpha * &p[i];
                r[i] -= &alpha * &Ap[i];
            }

            let rs_new = r
                .iter()
                .map(|v| (v * v).complete(v.prec()))
                .reduce(|a, b| a + b)
                .unwrap();
            if rs_new < Float::with_val(precision, 1e-30) {
                break;
            }
            let beta = (&rs_new / &rs_old).complete(rs_new.prec());
            for i in 0..p.len() {
                p[i] = &r[i] + (&beta * &p[i]).complete(beta.prec());
            }
            rs_old = rs_new;
        }
        x
    }

    /// LSQR algorithm to solve A x ≈ b
    pub fn lsqr(&self, b: &[Float], max_iters: usize) -> Vec<Float> {
        let precision = b[0].prec();
        let mut x = vec![Float::with_val(precision, 0); self.cols];
        let mut u = b.to_vec();
        let beta = Self::norm2_vec(&u);
        for ui in &mut u {
            *ui /= &beta;
        }

        let mut v = t!(self).mul(&u);
        let alpha = Self::norm2_vec(&v);
        for vi in &mut v {
            *vi /= &alpha;
        }

        let mut w = v.clone();
        let mut phibar = beta.clone();
        let mut rhobar = alpha.clone();

        for _ in 0..max_iters {
            let mut u_new = self.matmul_vec(&v);
            for (u_newi, ui) in u_new.iter_mut().zip(&u) {
                *u_newi -= alpha.clone() * ui;
            }
            let beta = Self::norm2_vec(&u_new);
            for ui in &mut u_new {
                *ui /= &beta;
            }
            u = u_new;

            let mut v_new = t!(self).mul(&u);
            for (v_newi, vi) in v_new.iter_mut().zip(&v) {
                *v_newi -= beta.clone() * vi;
            }
            let alpha = Self::norm2_vec(&v_new);
            for vi in &mut v_new {
                *vi /= &alpha;
            }
            v = v_new;

            let rho = (rhobar.clone().square() + beta.clone().square()).sqrt();
            let c = rhobar.clone() / &rho;
            let s = beta / &rho;
            let theta = s.clone() * &alpha;
            rhobar = -c.clone() * alpha;
            let phi = c * phibar.clone();
            phibar = s * phibar;

            for j in 0..x.len() {
                x[j] += &phi * &w[j];
            }
            for j in 0..w.len() {
                w[j] = &v[j] - (&theta * &w[j]).complete(theta.prec());
            }
        }

        x
    }

    /// Estimate the largest singular value (spectral norm) using power iteration
    pub fn spectral_norm_estimate(&self, max_iters: usize, tol: f64) -> Float {
        let precision = self.data[0].prec();
        let mut x = vec![Float::with_val(precision, 1); self.cols];
        let mut lambda = Float::with_val(precision, 0);

        for _ in 0..max_iters {
            let y = self.matmul_vec(&x);
            let norm_y = Self::norm2_vec(&y);
            for i in 0..self.cols {
                x[i] = y[i].clone() / &norm_y;
            }
            let y2 = self.matmul_vec(&x);
            let lambda_new = y
                .iter()
                .zip(&y2)
                .map(|(a, b)| (a * b).complete(a.prec()))
                .reduce(|a, b| a + b)
                .unwrap();
            if (&lambda_new - &lambda).complete(lambda_new.prec()).abs()
                < Float::with_val(precision, tol)
            {
                return lambda_new;
            }
            lambda.assign(lambda_new);
        }
        lambda
    }

    /// Estimate the smallest singular value using inverse power iteration with gradient solve
    pub fn smallest_singular_value_estimate(&self, max_iters: usize, tol: f64) -> Float {
        let precision = self.data[0].prec();
        let mut x = vec![Float::with_val(precision, 1); self.rows];
        let mut lambda = Float::with_val(precision, 0);

        for _ in 0..max_iters {
            let y = self.pseudo_inverse_solve(
                &x,
                20,
                PseudoInverseAlgorithm::GradientDescent {
                    alpha: Float::with_val(precision, 1e-3),
                },
            );
            let norm_y = Self::norm2_vec(&y);
            for i in 0..self.cols {
                x[i] = y[i].clone() / &norm_y;
            }
            let Ax = self.matmul_vec(&x);
            let lambda_new = x
                .iter()
                .zip(&Ax)
                .map(|(a, b)| (a * b).complete(a.prec()))
                .reduce(|a, b| a + b)
                .unwrap();
            if (&lambda_new - &lambda).complete(lambda_new.prec()).abs()
                < Float::with_val(precision, tol)
            {
                return lambda_new;
            }
            lambda.assign(lambda_new);
        }
        lambda
    }

    /// Estimate the condition number based on spectral and inverse estimates
    pub fn cond_estimate(&self, max_iters: usize, tol: f64) -> Float {
        let precision = self.data[0].prec();
        let sigma_max = self.spectral_norm_estimate(max_iters, tol);
        let sigma_min = self.smallest_singular_value_estimate(max_iters, tol);
        // TODO: Panic might not be the best way to handle this as singular matrices can totally occur
        if sigma_min.is_zero() {
            panic!("Singular matrix detected: smallest singular value is zero.");
        }

        Float::with_val(precision, &sigma_max / &sigma_min)
    }

    /// Estimate required bit precision to achieve `bits` of accurate output
    /// given a condition number estimate.
    ///
    /// Based on the fact that numerical error in solving Ax = b is amplified
    /// by the condition number κ = cond(A), we may lose up to log2(κ) bits of
    /// effective precision. To achieve `bits` of final result accuracy, we need:
    ///
    ///     required_bits = bits + ceil(log2(cond))
    ///
    /// This function returns that estimate.
    pub fn required_precision_for_cond(bits: usize, cond_estimate: &Float) -> usize {
        let extra = cond_estimate.clone().log2().ceil().to_integer().unwrap();
        bits + extra.to_usize_wrapping()
    }
}

impl RugMat {
    /*

    /// LU decomposition using Doolittle’s method (no pivoting)
    /// Returns (L, U) such that A = L * U
    pub fn lu_decompose(&self) -> (RugMat, RugMat) {
        assert_eq!(self.rows, self.cols, "Matrix must be square");
        let n = self.rows;
        let precision = self.data[0][0].precision();

        let mut l = RugMat::new(n, n, precision);
        let mut u = RugMat::new(n, n, precision);

        for i in 0..n {
            // U upper triangle
            for k in i..n {
                let mut sum = Float::with_val(precision, 0);
                for j in 0..i {
                    sum += &l.data[i][j] * &u.data[j][k];
                }
                u.data[i][k] = &self.data[i][k] - sum;
            }

            // L lower triangle
            for k in i..n {
                if i == k {
                    l.data[i][i] = Float::with_val(precision, 1);
                } else {
                    let mut sum = Float::with_val(precision, 0);
                    for j in 0..i {
                        sum += &l.data[k][j] * &u.data[j][i];
                    }
                    l.data[k][i] = (&self.data[k][i] - sum) / &u.data[i][i];
                }
            }
        }

        (l, u)
    }

    /// LU decomposition with partial pivoting.
    /// Returns (L, U, P) where P is a permutation vector: P[i] is the row index swapped into position i.
    pub fn lu_decompose_pivot(&self) -> (RugMat, RugMat, Vec<usize>) {
        assert_eq!(self.rows, self.cols, "Matrix must be square");
        let n = self.rows;
        let precision = self.data[0][0].precision();

        let mut a = self.data.clone();
        let mut p: Vec<usize> = (0..n).collect();

        for i in 0..n {
            // Partial pivoting: find row with max |a[j][i]|
            let mut max_row = i;
            let mut max_val = a[i][i].abs();

            for j in (i + 1)..n {
                let abs = a[j][i].abs();
                if abs > max_val {
                    max_val = abs.clone();
                    max_row = j;
                }
            }

            // Swap rows in `a` and permutation
            a.swap(i, max_row);
            p.swap(i, max_row);
        }

        let mut l = RugMat::new(n, n, precision);
        let mut u = RugMat::new(n, n, precision);

        for i in 0..n {
            for j in 0..n {
                if i <= j {
                    // Upper triangular matrix
                    let mut sum = Float::with_val(precision, 0);
                    for k in 0..i {
                        sum += &l.data[i][k] * &u.data[k][j];
                    }
                    u.data[i][j] = &a[i][j] - sum;
                } else {
                    // Lower triangular matrix
                    let mut sum = Float::with_val(precision, 0);
                    for k in 0..j {
                        sum += &l.data[i][k] * &u.data[k][j];
                    }
                    l.data[i][j] = (&a[i][j] - sum) / &u.data[j][j];
                }

                if i == j {
                    l.data[i][j] = Float::with_val(precision, 1);
                }
            }
        }

        (l, u, p)
    }

    /// Apply row permutation `p` to a vector (or right-hand side of Ax = b)
    pub fn apply_permutation(vec: &[Float], p: &[usize]) -> Vec<Float> {
        p.iter().map(|&i| vec[i].clone()).collect()
    }

    /// Solves Ly = b for y, where L is lower triangular.
    pub fn forward_substitution(&self, b: &[Float]) -> Vec<Float> {
        let n = self.rows;
        let precision = self.data[0][0].precision();
        let mut y = vec![Float::with_val(precision, 0); n];

        for i in 0..n {
            let mut sum = Float::with_val(precision, 0);
            for j in 0..i {
                sum += &self.data[i][j] * &y[j];
            }
            y[i] = (&b[i] - sum) / &self.data[i][i];
        }
        y
    }

    /// Solves Ux = y for x, where U is upper triangular.
    pub fn backward_substitution(&self, y: &[Float]) -> Vec<Float> {
        let n = self.rows;
        let precision = self.data[0][0].precision();
        let mut x = vec![Float::with_val(precision, 0); n];

        for i in (0..n).rev() {
            let mut sum = Float::with_val(precision, 0);
            for j in (i + 1)..n {
                sum += &self.data[i][j] * &x[j];
            }
            x[i] = (&y[i] - sum) / &self.data[i][i];
        }
        x
    }

    /// Solves Ax = b using LU decomposition with partial pivoting.
    pub fn solve(&self, b: &[Float]) -> Vec<Float> {
        let (l, u, p) = self.lu_decompose_pivot();
        let pb = RugMat::apply_permutation(b, &p);
        let y = l.forward_substitution(&pb);
        let x = u.backward_substitution(&y);
        x
    }


    /// Estimate condition number in infinity norm: ||A|| * ||A⁻¹|| (estimated)
    pub fn condition_number(&self) -> Float {
        let precision = self.data[0][0].precision();
        let n = self.rows;

        let norm_a = self.norm_inf();
        let mut a_inv_rowsum_max = Float::with_val(precision, 0);

        for i in 0..n {
            // Solve A x = e_i
            let mut e = vec![Float::with_val(precision, 0); n];
            e[i] = Float::with_val(precision, 1);

            let x = self.solve(&e); // Solve A⁻¹ · eᵢ

            let rowsum: Float = x
                .iter()
                .map(|v| v.clone().abs())
                .fold(Float::with_val(precision, 0), |acc, v| acc + v);

            if rowsum > a_inv_rowsum_max {
                a_inv_rowsum_max = rowsum;
            }
        }

        norm_a * a_inv_rowsum_max
    }

    /// Transpose of the matrix
    pub fn transpose(&self) -> RugMat {
        let precision = self.data[0][0].precision();
        let mut result = RugMat::new(self.cols, self.rows, precision);

        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j][i] = self.data[i][j].clone();
            }
        }
        result
    }

    /// Compute QR decomposition using Gram-Schmidt process
    pub fn qr_decompose(&self) -> (RugMat, RugMat) {
        let n = self.rows;
        let precision = self.data[0][0].precision();

        let mut q = RugMat::new(n, n, precision);
        let mut r = RugMat::new(n, n, precision);

        let mut a_cols: Vec<Vec<Float>> = (0..n)
            .map(|j| (0..n).map(|i| self.data[i][j].clone()).collect())
            .collect();

        let mut q_cols: Vec<Vec<Float>> = Vec::new();

        for j in 0..n {
            let mut v = a_cols[j].clone();
            for k in 0..j {
                let r_kj = dot(&q_cols[k], &a_cols[j], precision);
                for i in 0..n {
                    v[i] -= &q_cols[k][i] * &r_kj;
                }
                r.data[k][j] = r_kj;
            }

            let norm = norm2(&v, precision);
            for i in 0..n {
                v[i] /= &norm;
            }
            for i in 0..n {
                q.data[i][j] = v[i].clone();
            }
            r.data[j][j] = norm;
            q_cols.push(v);
        }

        (q, r)
    }

    /// Estimates eigenvalues of a symmetric matrix using QR iterations
    pub fn eigenvalues_via_qr(&self, max_iters: usize) -> Vec<Float> {
        let precision = self.data[0][0].precision();
        let mut a_k = self.clone();

        for _ in 0..max_iters {
            let (q, r) = a_k.qr_decompose();
            a_k = r.matmul(&q); // Aₖ₊₁ = Rₖ Qₖ
        }

        (0..self.rows).map(|i| a_k.data[i][i].clone()).collect()
    }

    /// Estimates singular values (abs of eigenvalues if symmetric)
    pub fn svd_symmetric(&self, max_iters: usize) -> Vec<Float> {
        let eigvals = self.eigenvalues_via_qr(max_iters);
        eigvals.into_iter().map(|λ| λ.abs()).collect()
    }

    /// Returns bidiagonal matrix B from A ≈ U * B * Vᵗ using Golub-Kahan
    pub fn bidiagonalize(&self) -> RugMat {
        let mut a = self.clone();
        let (m, n) = (self.rows, self.cols);
        let precision = self.data[0][0].precision();

        for i in 0..n.min(m) {
            // Householder from left (columns)
            let x: Vec<Float> = (i..m).map(|row| a.data[row][i].clone()).collect();
            let v = householder_vector(&x, precision);

            for col in i..n {
                let mut dot = Float::with_val(precision, 0);
                for row in i..m {
                    dot += &v[row - i] * &a.data[row][col];
                }
                for row in i..m {
                    a.data[row][col] -= &v[row - i] * &dot * 2.clone();
                }
            }

            if i < n - 2 {
                // Householder from right (rows)
                let x: Vec<Float> = (i + 1..n).map(|col| a.data[i][col].clone()).collect();
                let v = householder_vector(&x, precision);

                for row in i..m {
                    let mut dot = Float::with_val(precision, 0);
                    for col in i + 1..n {
                        dot += &v[col - i - 1] * &a.data[row][col];
                    }
                    for col in i + 1..n {
                        a.data[row][col] -= &v[col - i - 1] * &dot * 2.clone();
                    }
                }
            }
        }

        a // now bidiagonal
    }

    fn householder_vector(x: &[Float], precision: u32) -> Vec<Float> {
        let mut v = x.to_vec();
        let norm_x = norm2(x, precision);
        let sign = if x[0].significant_bits() == 0
            || x[0].significant_bits() > 0 && x[0].is_sign_negative()
        {
            -Float::with_val(precision, 1)
        } else {
            Float::with_val(precision, 1)
        };

        v[0] += &sign * &norm_x;
        let norm_v = norm2(&v, precision);
        for val in &mut v {
            *val /= &norm_v;
        }

        v
    }

    pub fn svd(&self, max_iters: usize) -> Vec<Float> {
        let precision = self.data[0][0].precision();
        let mut b = self.bidiagonalize();
        let mut bt = b.transpose();
        let mut btb = bt.matmul(&b);

        for _ in 0..max_iters {
            let (q, r) = btb.qr_decompose();
            btb = r.matmul(&q);
        }

        (0..btb.rows)
            .map(|i| btb.data[i][i].clone().sqrt())
            .collect()
    }

    pub fn full_svd(&self, max_iters: usize) -> SVD {
        let precision = self.data[0][0].precision();
        let at = self.transpose();
        let ata = at.matmul(self);
        let aat = self.matmul(&at);

        // Step 1: eigendecomposition
        let eigvals_v = ata.eigenvalues_via_qr(max_iters);
        let mut svals: Vec<Float> = eigvals_v.iter().map(|λ| λ.clone().sqrt()).collect();

        // Step 2: right singular vectors (columns of V)
        let mut v_cols = Vec::new();
        for λ in &eigvals_v {
            let v = power_method(&ata, λ.clone(), max_iters);
            v_cols.push(v);
        }

        let vt = RugMat::from_columns(&v_cols);

        // Step 3: left singular vectors
        let mut u_cols = Vec::new();
        for (i, s) in svals.iter().enumerate() {
            if s.significant_bits() == 0 {
                continue;
            } // avoid division by 0
            let av = self.matmul_vec(&vt.get_column(i));
            let ui: Vec<Float> = av.into_iter().map(|x| x / s).collect();
            u_cols.push(ui);
        }

        let u = RugMat::from_columns(&u_cols);

        SVD { u, s: svals, vt }
    }

    pub fn get_column(&self, j: usize) -> Vec<Float> {
        (0..self.rows).map(|i| self.data[i][j].clone()).collect()
    }

    pub fn from_columns(cols: &[Vec<Float>]) -> RugMat {
        let rows = cols[0].len();
        let cols_len = cols.len();
        let data = (0..rows)
            .map(|i| (0..cols_len).map(|j| cols[j][i].clone()).collect())
            .collect();
        RugMat {
            data,
            rows,
            cols: cols_len,
        }
    }

    pub fn matmul_vec(&self, v: &[Float]) -> Vec<Float> {
        let precision = self.data[0][0].precision();
        (0..self.rows)
            .map(|i| {
                (0..self.cols)
                    .map(|j| &self.data[i][j] * &v[j])
                    .fold(Float::with_val(precision, 0), |acc, x| acc + x)
            })
            .collect()
    }

    fn power_method(a: &RugMat, λ: Float, max_iter: usize) -> Vec<Float> {
        let precision = λ.precision();
        let mut x = vec![Float::with_val(precision, 1); a.cols];

        for _ in 0..max_iter {
            x = a.matmul_vec(&x);
            let norm = norm2(&x, precision);
            for xi in &mut x {
                *xi /= &norm;
            }
        }

        x
    }
    pub fn transpose_data(data: &[Vec<Float>]) -> Vec<Vec<Float>> {
        let rows = data.len();
        let cols = data[0].len();
        (0..cols)
            .map(|j| (0..rows).map(|i| data[i][j].clone()).collect())
            .collect()
    }

    pub fn pseudoinverse(&self, threshold: &Float, max_iters: usize) -> RugMat {
        let precision = threshold.precision();
        let svd = self.full_svd(max_iters);

        // Invert non-zero singular values
        let sigma_inv = svd
            .s
            .iter()
            .map(|s| {
                if s > threshold {
                    Float::with_val(precision, 1) / s
                } else {
                    Float::with_val(precision, 0)
                }
            })
            .collect::<Vec<_>>();

        // Build Σ⁺: diag(sigma_inv) as matrix
        let mut sigma_inv_mat = RugMat::new(sigma_inv.len(), sigma_inv.len(), precision);
        for i in 0..sigma_inv.len() {
            sigma_inv_mat.data[i][i] = sigma_inv[i].clone();
        }

        // A⁺ = V · Σ⁺ · Uᵗ
        let v = svd.vt.transpose();
        let u_t = svd.u.transpose();

        let v_sigma = v.matmul(&sigma_inv_mat);
        let a_pseudo = v_sigma.matmul(&u_t);
        a_pseudo
    }

    pub fn solve_least_squares(
        &self,
        b: &[Float],
        threshold: &Float,
        max_iters: usize,
    ) -> Vec<Float> {
        let pseudo_inv = self.pseudoinverse(threshold, max_iters);
        pseudo_inv.matmul_vec(b)
    }

    fn power_iteration(a: &RugMat, max_iters: usize) -> Float {
        let n = a.rows;
        let precision = a.data[0][0].precision();
        let mut x = vec![Float::with_val(precision, 1); n];

        for _ in 0..max_iters {
            x = a.matmul_vec(&x);
            let norm = norm2(&x, precision);
            for xi in &mut x {
                *xi /= &norm;
            }
        }

        // Rayleigh quotient
        let ax = a.matmul_vec(&x);
        let num = dot(&x, &ax, precision);
        let den = dot(&x, &x, precision);

        num / den
    }

    fn estimate_condition_number(a: &RugMat, max_iters: usize) -> Float {
        let at = a.transpose();
        let ata = at.matmul(a);

        let lambda_max = power_iteration(&ata, max_iters);
        let inv_ata = ata.pseudoinverse(
            &Float::with_val(ata.data[0][0].precision(), 1e-50),
            max_iters,
        );
        let lambda_min = power_iteration(&inv_ata, max_iters);

        (lambda_max.clone().sqrt()) * (lambda_min.clone().sqrt())
    }

    fn inverse_iteration(a: &RugMat, max_iters: usize) -> Float {
        let n = a.rows;
        let precision = a.data[0][0].precision();
        let mut x = vec![Float::with_val(precision, 1); n];

        for _ in 0..max_iters {
            // Solve A y = x
            let y = a.solve(&x); // Uses LU solve
            let norm = norm2(&y, precision);

            for i in 0..n {
                x[i] = &y[i] / &norm;
            }
        }

        // Rayleigh quotient estimate
        let ax = a.matmul_vec(&x);
        let num = dot(&x, &ax, precision);
        let den = dot(&x, &x, precision);
        num / den
    }

    pub fn estimate_condition_number_auto(&self, max_iters: usize, max_bits: u32) -> (Float, u32) {
        let mut bits = 64;
        let mut last_estimate = Float::with_val(bits, 0);

        while bits <= max_bits {
            // Clone matrix at new precision
            let mut a_precise = self.clone();
            for row in &mut a_precise.data {
                for val in row {
                    *val = Float::with_val(bits, val);
                }
            }

            // Compute AᵗA for general matrix
            let ata = a_precise.transpose().matmul(&a_precise);

            // Power and inverse iteration
            let lambda_max = power_iteration(&ata, max_iters);
            let lambda_min = inverse_iteration(&ata, max_iters);

            if lambda_min.significant_bits() == 0 {
                bits *= 2;
                continue; // likely numerical underflow
            }

            let cond = Float::with_val(bits, &lambda_max / &lambda_min).sqrt();

            // Check for convergence
            if (&cond - &last_estimate).abs() < Float::with_val(bits, 1e-10) {
                return (cond, bits);
            }

            last_estimate = cond;
            bits *= 2;
        }

        (last_estimate, bits)
    }

    pub fn from_faer_f64(mat: &faer::Mat<f64>, precision: u32) -> Self {
        let rows = mat.nrows();
        let cols = mat.ncols();

        let mut data = vec![vec![Float::with_val(precision, 0); cols]; rows];

        for i in 0..rows {
            for j in 0..cols {
                data[i][j] = Float::with_val(precision, mat.read(i, j));
            }
        }

        RugMat { data, rows, cols }
    }

    pub fn from_faer_f32(mat: &faer::Mat<f32>, precision: u32) -> Self {
        let rows = mat.nrows();
        let cols = mat.ncols();

        let mut data = vec![vec![Float::with_val(precision, 0); cols]; rows];

        for i in 0..rows {
            for j in 0..cols {
                data[i][j] = Float::with_val(precision, mat.read(i, j) as f64);
            }
        }

        RugMat { data, rows, cols }
    }

    */
    pub fn trace_norm_approx(&self, max_iters: usize, tol: f64, max_singulars: usize) -> Float {
        let precision = self.data[0].prec();
        let mut A = self.clone();
        let mut total = Float::with_val(precision, 0);

        for _ in 0..max_singulars {
            let sigma = A.spectral_norm_estimate(max_iters, tol);
            if sigma < Float::with_val(precision, tol) {
                break;
            }
            total += &sigma;

            // Deflate the dominant singular value component
            // (Not full deflation, approximate)
            let mut x = vec![Float::with_val(precision, 1); A.cols];
            for _ in 0..max_iters {
                let y = A.matmul_vec(&x);
                let y2 = Transpose::new(&A).mul(&y);
                let norm = RugMat::norm2_vec(&y2);
                if norm < Float::with_val(precision, tol) {
                    break;
                }
                for i in 0..x.len() {
                    x[i] = y2[i].clone() / &norm;
                }
            }

            let y = A.matmul_vec(&x);
            for i in 0..A.rows {
                for j in 0..A.cols {
                    let mut temp = (&y[i] * &x[j]).complete(y[i].prec());
                    temp *= &sigma;
                    A[(i, j)] -= temp;
                }
            }
        }
        total
    }

    pub fn max_entry_norm(&self) -> Float {
        let precision = self.data[0].prec();
        let mut max_val = Float::with_val(precision, 0);
        for j in 0..self.cols {
            for i in 0..self.rows {
                let val = self[(i, j)].clone().abs();
                if val > max_val {
                    max_val = val;
                }
            }
        }
        max_val
    }

    pub fn lp_norm(&self, p: f64, epsilon: Option<f64>) -> Float {
        assert!(p > 0.0 && p <= 2.0, "Lp norm only defined for p in (0, 2]");

        let precision = self.data[0].prec();
        let mut acc = Float::with_val(precision * 2, 0);
        let eps = epsilon.unwrap_or(0.0);

        for j in 0..self.cols {
            for i in 0..self.rows {
                let mut val = self[(i, j)].clone().abs();
                if eps > 0.0 && val < Float::with_val(precision, eps) {
                    continue;
                }
                val = val.pow(p);
                acc += val;
            }
        }

        acc.pow(1.0 / p)
    }

    pub fn l0_norm(&self) -> usize {
        let mut count = 0;
        for j in 0..self.cols {
            for i in 0..self.rows {
                if !self[(i, j)].is_zero() {
                    count += 1;
                }
            }
        }
        count
    }

    pub fn frobenius_norm(&self) -> Float {
        let precision = self.data[0].prec();
        let mut acc = Float::with_val(precision * 2, 0);
        for j in 0..self.cols {
            for i in 0..self.rows {
                acc += self[(i, j)].clone().square();
            }
        }
        acc.sqrt()
    }

    /// 1-norm (maximum absolute column sum)
    pub fn norm1(&self) -> Float {
        let precision = self.data[0].prec();
        let mut col_sums = vec![Float::with_val(precision, 0); self.cols];

        for j in 0..self.cols {
            for i in 0..self.rows {
                col_sums[j] += self[(i, j)].clone().abs();
            }
        }

        col_sums
            .into_iter()
            .reduce(|a, b| if a > b { a } else { b })
            .unwrap()
    }

    pub fn identity(size: usize, precision: u32) -> Self {
        let value = Float::with_val(precision, 1);
        let mut mat = RugMat::new(size, size, precision);
        for i in 0..size {
            mat[(i, i)] = value.clone();
        }
        mat
    }

    pub fn diagonal_from_f64(diag: &[f64], precision: u32) -> Self {
        let mut mat = RugMat::new(diag.len(), diag.len(), precision);
        for (i, &v) in diag.iter().enumerate() {
            mat[(i, i)] = Float::with_val(precision, v);
        }
        mat
    }

    pub fn from_vecvec(vecvec: Vec<Vec<Float>>) -> Self {
        let rows = vecvec.len();
        let cols = vecvec[0].len();
        let precision = vecvec[0][0].prec();
        let mut mat = RugMat::new(rows, cols, precision);
        for (i, row) in vecvec.into_iter().enumerate() {
            for (j, val) in row.into_iter().enumerate() {
                mat[(i, j)] = val;
            }
        }
        mat
    }
}

#[test]
fn test_power_iteration_identity() {
    let mat = RugMat::identity(5, 128);
    let sigma = mat.spectral_norm_estimate(100, 1e-20);
    assert!(sigma.to_f64() - 1.0 < 1e-10);
}

#[test]
fn test_power_iteration_diagonal() {
    let diag_values = vec![1.0, 2.0, 0.5, 4.0, 3.0];
    let mat = RugMat::diagonal_from_f64(&diag_values, 128);
    let sigma = mat.spectral_norm_estimate(500, 1e-20);
    println!("Sigma: {}", sigma);
    assert!((sigma.to_f64() - 4.0).abs() < 1e-10);
}

#[test]
fn test_norms_simple_matrix() {
    let data = vec![
        vec![Float::with_val(128, 1), Float::with_val(128, 2)],
        vec![Float::with_val(128, 3), Float::with_val(128, 4)],
    ];
    let mat = RugMat::from_vecvec(data);

    let norm1 = mat.norm1();
    let norm_inf = mat.norm_inf();

    assert!((norm1.to_f64() - 6.0).abs() < 1e-10); // max column sum
    assert!((norm_inf.to_f64() - 7.0).abs() < 1e-10); // max row sum
}

#[test]
fn test_matmul_identity() {
    let precision = 128;
    let mat = RugMat::from_vecvec(vec![
        vec![Float::with_val(precision, 1), Float::with_val(precision, 2)],
        vec![Float::with_val(precision, 3), Float::with_val(precision, 4)],
    ]);
    let id = RugMat::identity(2, precision);
    let result = mat.matmul(&id);
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (result[(i, j)].clone() - mat[(i, j)].clone()).abs()
                    < Float::with_val(precision, 1e-20)
            );
        }
    }
}

#[test]
fn test_matmul_vec_identity() {
    let precision = 128;
    let id = RugMat::identity(3, precision);
    let vec = vec![
        Float::with_val(precision, 1),
        Float::with_val(precision, 2),
        Float::with_val(precision, 3),
    ];
    let result = id.matmul_vec(&vec);
    for i in 0..3 {
        assert!((result[i].clone() - vec[i].clone()).abs() < Float::with_val(precision, 1e-20));
    }
}

#[test]
fn test_matmul_known() {
    let precision = 128;
    let a = RugMat::from_vecvec(vec![
        vec![Float::with_val(precision, 1), Float::with_val(precision, 2)],
        vec![Float::with_val(precision, 0), Float::with_val(precision, 1)],
    ]);
    let b = RugMat::from_vecvec(vec![
        vec![Float::with_val(precision, 3), Float::with_val(precision, 4)],
        vec![Float::with_val(precision, 5), Float::with_val(precision, 6)],
    ]);
    let result = a.matmul(&b);

    let expected = vec![
        vec![
            Float::with_val(precision, 13), // 1*3 + 2*5
            Float::with_val(precision, 16), // 1*4 + 2*6
        ],
        vec![
            Float::with_val(precision, 5), // 0*3 + 1*5
            Float::with_val(precision, 6), // 0*4 + 1*6
        ],
    ];
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (result[(i, j)].clone() - expected[i][j].clone()).abs()
                    < Float::with_val(precision, 1e-20)
            );
        }
    }
}

#[test]
fn test_matmul_vec_known() {
    let precision = 128;
    let mat = RugMat::from_vecvec(vec![
        vec![Float::with_val(precision, 2), Float::with_val(precision, 0)],
        vec![Float::with_val(precision, 1), Float::with_val(precision, 3)],
    ]);
    let vec = vec![Float::with_val(precision, 1), Float::with_val(precision, 2)];
    let result = mat.matmul_vec(&vec);

    let expected = vec![
        Float::with_val(precision, 2), // 2*1 + 0*2
        Float::with_val(precision, 7), // 1*1 + 3*2
    ];
    for i in 0..2 {
        assert!(
            (result[i].clone() - expected[i].clone()).abs() < Float::with_val(precision, 1e-20)
        );
    }
}
