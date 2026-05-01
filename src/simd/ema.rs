//! SIMD-accelerated exponential moving average over a slice.
//!
//! EMA is a first-order recurrence — `s_t = k*x_t + (1-k)*s_{t-1}` —
//! which naively forbids vectorization. We get around this with the
//! standard closed-form trick: for a block of 4 inputs starting from a
//! known `s_prev`, every output in the block can be expressed as a
//! linear combination of the block's inputs and `s_prev`. With
//! `a = 1 - k`:
//!
//! ```text
//! [s_0]   [ 1     0    0    0 ] [x_0]              [a  ]
//! [s_1] = [ a     1    0    0 ] [x_1] * k +  s_prev * [a^2]
//! [s_2]   [ a^2   a    1    0 ] [x_2]              [a^3]
//! [s_3]   [ a^3   a^2  a    1 ] [x_3]              [a^4]
//! ```
//!
//! After processing a block, `s_prev` becomes the last lane of the
//! result. Inter-block dependency stays serial (one f64 per block) but
//! the matrix-vector product is fully vectorized — about 2x speedup on
//! AVX2 over the scalar recurrence.
//!
//! Bootstrap convention: `out[0] = values[0]` (no smoothing on the
//! first sample) — this matches the streaming `ExponentialMovingAverage`
//! indicator's `is_new` initialization.

use wide::f64x4;

const LANES: usize = 4;

/// Allocates and returns the EMA of `values` with smoothing factor `k`.
/// The first output is `values[0]` (matching the streaming indicator).
pub fn ema(values: &[f64], k: f64) -> Vec<f64> {
    let mut out = vec![0.0; values.len()];
    ema_into(values, k, &mut out);
    out
}

/// Writes the EMA into `out`. Panics if lengths disagree.
pub fn ema_into(values: &[f64], k: f64, out: &mut [f64]) {
    assert_eq!(
        values.len(),
        out.len(),
        "ema_into: out.len() must equal values.len()"
    );
    if values.is_empty() {
        return;
    }
    out[0] = values[0];
    if values.len() == 1 {
        return;
    }
    ema_continuation_into(&values[1..], k, values[0], &mut out[1..]);
}

/// Continuation EMA: given a prior state `s_prev`, compute EMA outputs
/// for `values` and write them into `out`. Used by stateful indicators
/// that already have an EMA running and want to extend it with a batch.
pub fn ema_continuation_into(values: &[f64], k: f64, s_prev: f64, out: &mut [f64]) {
    assert_eq!(
        values.len(),
        out.len(),
        "ema_continuation_into: out.len() must equal values.len()"
    );
    if values.is_empty() {
        return;
    }

    let a = 1.0 - k;

    // Precompute powers of a and the column vectors for the
    // lower-triangular block matrix.
    let a2 = a * a;
    let a3 = a2 * a;
    let a4 = a3 * a;

    let col0 = f64x4::new([1.0, a, a2, a3]);
    let col1 = f64x4::new([0.0, 1.0, a, a2]);
    let col2 = f64x4::new([0.0, 0.0, 1.0, a]);
    let col3 = f64x4::new([0.0, 0.0, 0.0, 1.0]);
    let a_powers = f64x4::new([a, a2, a3, a4]);
    let k_vec = f64x4::splat(k);

    let mut s_prev = s_prev;
    let n = values.len();
    let mut i = 0;

    while i + LANES <= n {
        let x0 = f64x4::splat(values[i]);
        let x1 = f64x4::splat(values[i + 1]);
        let x2 = f64x4::splat(values[i + 2]);
        let x3 = f64x4::splat(values[i + 3]);

        let xc = x0 * col0 + x1 * col1 + x2 * col2 + x3 * col3;
        let block = k_vec * xc + a_powers * f64x4::splat(s_prev);

        let arr = block.to_array();
        out[i] = arr[0];
        out[i + 1] = arr[1];
        out[i + 2] = arr[2];
        out[i + 3] = arr[3];

        s_prev = arr[3];
        i += LANES;
    }

    // Scalar tail (0–3 elements).
    while i < n {
        s_prev = k * values[i] + a * s_prev;
        out[i] = s_prev;
        i += 1;
    }
}

#[cfg(test)]
pub(crate) fn ema_scalar(values: &[f64], k: f64) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![0.0; n];
    if n == 0 {
        return out;
    }
    out[0] = values[0];
    let a = 1.0 - k;
    let mut s = values[0];
    for i in 1..n {
        s = k * values[i] + a * s;
        out[i] = s;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_slice_eq(a: &[f64], b: &[f64], rel: f64, abs: f64) {
        assert_eq!(a.len(), b.len());
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
        // k=0.5: s_0=2, s_1=0.5*5+0.5*2=3.5, s_2=0.5*1+0.5*3.5=2.25, s_3=0.5*6.25+0.5*2.25=4.25
        let v = [2.0, 5.0, 1.0, 6.25];
        let out = ema(&v, 0.5);
        let expected = [2.0, 3.5, 2.25, 4.25];
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12, "got {} expected {}", a, b);
        }
    }

    #[test]
    fn empty_and_single() {
        assert_eq!(ema(&[], 0.5), Vec::<f64>::new());
        assert_eq!(ema(&[3.14], 0.5), vec![3.14]);
    }

    #[test]
    fn parity_random() {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 16, 17, 31, 100, 1000, 10_000] {
            let values: Vec<f64> = (0..n)
                .map(|i| 100.0 + ((i as f64) * 0.07).sin() * 10.0)
                .collect();
            for k in [0.01, 0.1, 0.0645, 0.5, 0.9] {
                let s = ema_scalar(&values, k);
                let v = ema(&values, k);
                // Closed-form SIMD vs scalar recurrence: rounding may
                // differ by a few ULPs per block. 1e-10 relative is
                // very tight for f64 — drift bigger than that means
                // the math is wrong.
                approx_slice_eq(&v, &s, 1e-10, 1e-12);
            }
        }
    }

    #[test]
    fn matches_streaming_indicator() {
        use crate::indicators::ExponentialMovingAverage;
        use crate::Next;
        use chrono::{TimeZone, Utc};
        use std::time::Duration;

        // 30-period EMA — k = 2/(30+1) = 0.06451612...
        let mut streaming =
            ExponentialMovingAverage::new(Duration::from_secs(30 * 86400)).unwrap();
        let start = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let values: Vec<f64> = (0..200).map(|i| 50.0 + (i as f64) * 0.3).collect();

        let streaming_out: Vec<f64> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| streaming.next((start + chrono::Duration::days(i as i64), v)))
            .collect();

        let k = 2.0 / 31.0;
        let batch_out = ema(&values, k);

        for (i, (s, b)) in streaming_out.iter().zip(batch_out.iter()).enumerate() {
            let diff = (s - b).abs();
            assert!(
                diff <= 1e-10 * s.abs().max(b.abs()).max(1e-12),
                "index {}: streaming={} batch={}",
                i,
                s,
                b
            );
        }
    }

    #[test]
    fn continuation_resumes_correctly() {
        // Splitting an EMA into prefix + continuation must give the
        // same result as one batch call.
        let values: Vec<f64> = (0..50).map(|i| 10.0 + (i as f64).sin()).collect();
        let k = 0.2;

        let full = ema(&values, k);

        let split = 7;
        let prefix = ema(&values[..split], k);
        let s_prev = *prefix.last().unwrap();
        let mut cont = vec![0.0; values.len() - split];
        ema_continuation_into(&values[split..], k, s_prev, &mut cont);

        let mut joined = prefix;
        joined.extend(cont);

        for (i, (a, b)) in full.iter().zip(joined.iter()).enumerate() {
            assert!((a - b).abs() < 1e-12, "index {}: full={} joined={}", i, a, b);
        }
    }
}
