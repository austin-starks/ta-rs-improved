//! SIMD reductions over a full slice: sum, sum-of-squares, mean,
//! variance, std-dev.

use wide::f64x4;

const LANES: usize = 4;

/// SIMD sum of `values`. Splits into 4-lane chunks, accumulates four
/// independent partial sums in parallel, then horizontally reduces.
///
/// Floating-point addition is not associative, so this can differ from a
/// strict left-to-right scalar sum by ~1 ULP per element. Parity tests
/// use a relative tolerance, not bit-exact equality.
pub fn sum(values: &[f64]) -> f64 {
    let chunks = values.chunks_exact(LANES);
    let remainder = chunks.remainder();

    let mut acc = f64x4::ZERO;
    for chunk in chunks {
        let v = f64x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]);
        acc += v;
    }

    let lanes = acc.to_array();
    let mut total = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for &x in remainder {
        total += x;
    }
    total
}

/// SIMD sum of squares.
pub fn sum_squares(values: &[f64]) -> f64 {
    let chunks = values.chunks_exact(LANES);
    let remainder = chunks.remainder();

    let mut acc = f64x4::ZERO;
    for chunk in chunks {
        let v = f64x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]);
        acc += v * v;
    }

    let lanes = acc.to_array();
    let mut total = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for &x in remainder {
        total += x * x;
    }
    total
}

/// Arithmetic mean. Returns `0.0` for an empty slice (matching the
/// streaming indicators' behavior on an empty window).
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    sum(values) / values.len() as f64
}

/// Population variance. Computed as `E[X^2] - E[X]^2` using two
/// vectorized passes' worth of accumulators in a single pass.
///
/// This is the same formulation as `StandardDeviation::next` so callers
/// computing stats on a closed window get matching values.
pub fn variance(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }

    let chunks = values.chunks_exact(LANES);
    let remainder = chunks.remainder();

    let mut sum_acc = f64x4::ZERO;
    let mut sq_acc = f64x4::ZERO;
    for chunk in chunks {
        let v = f64x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]);
        sum_acc += v;
        sq_acc += v * v;
    }

    let s_lanes = sum_acc.to_array();
    let q_lanes = sq_acc.to_array();
    let mut s = s_lanes[0] + s_lanes[1] + s_lanes[2] + s_lanes[3];
    let mut q = q_lanes[0] + q_lanes[1] + q_lanes[2] + q_lanes[3];
    for &x in remainder {
        s += x;
        q += x * x;
    }

    let n_f = n as f64;
    let mean = s / n_f;
    (q - s * mean) / n_f
}

/// Population standard deviation.
pub fn std_dev(values: &[f64]) -> f64 {
    variance(values).sqrt()
}

// --- Scalar reference implementations (used in parity tests) ---

#[cfg(test)]
pub(crate) fn sum_scalar(values: &[f64]) -> f64 {
    let mut s = 0.0;
    for &x in values {
        s += x;
    }
    s
}

#[cfg(test)]
pub(crate) fn sum_squares_scalar(values: &[f64]) -> f64 {
    let mut s = 0.0;
    for &x in values {
        s += x * x;
    }
    s
}

#[cfg(test)]
pub(crate) fn variance_scalar(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    let s = sum_scalar(values);
    let q = sum_squares_scalar(values);
    let mean = s / n as f64;
    (q - s * mean) / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, rel: f64, abs: f64) -> bool {
        let diff = (a - b).abs();
        diff <= abs || diff <= rel * a.abs().max(b.abs())
    }

    #[test]
    fn empty_slice() {
        assert_eq!(sum(&[]), 0.0);
        assert_eq!(sum_squares(&[]), 0.0);
        assert_eq!(mean(&[]), 0.0);
        assert_eq!(variance(&[]), 0.0);
        assert_eq!(std_dev(&[]), 0.0);
    }

    #[test]
    fn known_values() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(sum(&v), 10.0);
        assert_eq!(sum_squares(&v), 30.0);
        assert_eq!(mean(&v), 2.5);
        // population variance of [1,2,3,4] = 1.25
        assert!(approx_eq(variance(&v), 1.25, 1e-12, 1e-12));
    }

    #[test]
    fn parity_sum_random() {
        // pseudo-random deterministic input across non-multiple-of-4 sizes
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 16, 17, 31, 64, 100, 1000, 10_000] {
            let values: Vec<f64> = (0..n)
                .map(|i| ((i as f64) * 0.37182).sin() * 1e3)
                .collect();
            let s = sum(&values);
            let r = sum_scalar(&values);
            assert!(
                approx_eq(s, r, 1e-12, 1e-9),
                "sum mismatch at n={}: simd={} scalar={}",
                n,
                s,
                r
            );

            let sq = sum_squares(&values);
            let rq = sum_squares_scalar(&values);
            assert!(
                approx_eq(sq, rq, 1e-12, 1e-9),
                "sum_squares mismatch at n={}: simd={} scalar={}",
                n,
                sq,
                rq
            );

            let v = variance(&values);
            let rv = variance_scalar(&values);
            assert!(
                approx_eq(v, rv, 1e-10, 1e-9),
                "variance mismatch at n={}: simd={} scalar={}",
                n,
                v,
                rv
            );
        }
    }

    #[test]
    fn matches_standard_deviation_indicator() {
        // Sanity check: stddev over a fixed slice should match what the
        // streaming StandardDeviation indicator produces after consuming
        // the same values within a single window.
        use crate::indicators::StandardDeviation;
        use crate::Next;
        use chrono::{TimeZone, Utc};
        use std::time::Duration;

        let values = [10.0, 20.0, 30.0, 20.0, 10.0];
        let mut sd =
            StandardDeviation::new(Duration::from_secs(60 * 60 * 24 * 365)).unwrap();
        let start = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let mut last = 0.0;
        for (i, &v) in values.iter().enumerate() {
            last = sd.next((start + chrono::Duration::days(i as i64), v));
        }
        let batch = std_dev(&values);
        assert!(
            approx_eq(last, batch, 1e-10, 1e-9),
            "streaming={} batch={}",
            last,
            batch
        );
    }
}
