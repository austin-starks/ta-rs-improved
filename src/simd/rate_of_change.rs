//! Lagged rate-of-change over a slice.
//!
//! For each `i >= lag`, computes `(values[i] - values[i - lag]) /
//! values[i - lag] * 100`. For `i < lag`, writes `0.0` (consistent with
//! how the streaming `RateOfChange` indicator behaves before its window
//! has filled). Division by zero is handled by writing `0.0` for that
//! element, again matching the streaming indicator.
//!
//! This is the pure-array form of ROC and the easiest SIMD win in the
//! library — every output is independent of every other output.

use wide::{f64x4, CmpEq};

const LANES: usize = 4;

/// Allocates and returns the lagged ROC of `values`. See module docs.
///
/// `lag` must be >= 1; passing 0 returns an all-zero vector to match the
/// streaming behavior on a degenerate input.
pub fn rate_of_change(values: &[f64], lag: usize) -> Vec<f64> {
    let mut out = vec![0.0; values.len()];
    rate_of_change_into(values, lag, &mut out);
    out
}

/// Writes the lagged ROC into `out`. `out.len()` must equal
/// `values.len()`.
pub fn rate_of_change_into(values: &[f64], lag: usize, out: &mut [f64]) {
    assert_eq!(
        values.len(),
        out.len(),
        "rate_of_change_into: out.len() must equal values.len()"
    );

    let n = values.len();
    if lag == 0 || n <= lag {
        for slot in out.iter_mut() {
            *slot = 0.0;
        }
        return;
    }

    // Leading `lag` elements have no predecessor — emit zeros.
    for slot in out[..lag].iter_mut() {
        *slot = 0.0;
    }

    let one_hundred = f64x4::splat(100.0);
    let zero = f64x4::ZERO;

    let mut i = lag;
    // SIMD body: process 4 outputs at a time.
    while i + LANES <= n {
        let cur = f64x4::new([
            values[i],
            values[i + 1],
            values[i + 2],
            values[i + 3],
        ]);
        let prev = f64x4::new([
            values[i - lag],
            values[i + 1 - lag],
            values[i + 2 - lag],
            values[i + 3 - lag],
        ]);

        let diff = cur - prev;
        let ratio = diff / prev * one_hundred;
        // Where prev == 0, emit 0 instead of NaN/inf.
        let safe = prev.cmp_eq(zero).blend(zero, ratio);
        let arr = safe.to_array();
        out[i] = arr[0];
        out[i + 1] = arr[1];
        out[i + 2] = arr[2];
        out[i + 3] = arr[3];
        i += LANES;
    }

    // Scalar tail.
    while i < n {
        let prev = values[i - lag];
        out[i] = if prev == 0.0 {
            0.0
        } else {
            (values[i] - prev) / prev * 100.0
        };
        i += 1;
    }
}

#[cfg(test)]
pub(crate) fn rate_of_change_scalar(values: &[f64], lag: usize) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![0.0; n];
    if lag == 0 || n <= lag {
        return out;
    }
    for i in lag..n {
        let prev = values[i - lag];
        out[i] = if prev == 0.0 {
            0.0
        } else {
            (values[i] - prev) / prev * 100.0
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_slice_eq(a: &[f64], b: &[f64], rel: f64, abs: f64) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            let tol = abs.max(rel * x.abs().max(y.abs()));
            assert!(
                diff <= tol,
                "mismatch at index {}: simd={} scalar={} diff={}",
                i,
                x,
                y,
                diff
            );
        }
    }

    #[test]
    fn known_values() {
        // values: 100, 110, 121, 133.1 → 10% growth each step
        let v = vec![100.0, 110.0, 121.0, 133.1];
        let out = rate_of_change(&v, 1);
        assert_eq!(out[0], 0.0);
        for i in 1..v.len() {
            assert!((out[i] - 10.0).abs() < 1e-9, "out[{}] = {}", i, out[i]);
        }
    }

    #[test]
    fn lag_larger_than_input() {
        let v = vec![1.0, 2.0, 3.0];
        assert_eq!(rate_of_change(&v, 5), vec![0.0, 0.0, 0.0]);
        assert_eq!(rate_of_change(&v, 3), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn zero_predecessor_yields_zero() {
        let v = vec![0.0, 10.0, 0.0, 5.0, 0.0, 7.0];
        let out = rate_of_change(&v, 2);
        // i=2: prev=0 → 0
        // i=3: prev=10 → (5-10)/10 * 100 = -50
        // i=4: prev=0 → 0
        // i=5: prev=5 → (7-5)/5 * 100 = 40
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 0.0);
        assert!((out[3] - -50.0).abs() < 1e-9);
        assert_eq!(out[4], 0.0);
        assert!((out[5] - 40.0).abs() < 1e-9);
    }

    #[test]
    fn parity_random_sizes_and_lags() {
        for n in [0, 1, 4, 5, 7, 8, 16, 31, 100, 1000, 5000] {
            let values: Vec<f64> = (0..n)
                .map(|i| 100.0 + ((i as f64) * 0.13).sin() * 5.0)
                .collect();
            for lag in [1, 2, 3, 4, 5, 7, 16, 32] {
                let s = rate_of_change_scalar(&values, lag);
                let v = rate_of_change(&values, lag);
                approx_slice_eq(&v, &s, 1e-12, 1e-12);
            }
        }
    }

    #[test]
    #[should_panic(expected = "out.len()")]
    fn into_length_mismatch_panics() {
        let v = [1.0, 2.0, 3.0];
        let mut out = [0.0; 2];
        rate_of_change_into(&v, 1, &mut out);
    }
}
