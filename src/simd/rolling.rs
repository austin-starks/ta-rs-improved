//! Rolling (windowed) mean and standard deviation over a slice.
//!
//! For each output index `i`, the statistic is computed over the
//! trailing window `values[i + 1 - w ..= i]`. Indices `i < w - 1`
//! produce a partial-window result over `values[..= i]`, which mirrors
//! how the streaming `SimpleMovingAverage` and `StandardDeviation`
//! indicators behave before their windows fill.
//!
//! The hot path is a running-sum subtract-add (O(n) total work
//! regardless of window size). SIMD is applied to the warm-up sum and
//! to the initial sum-of-squares, where vectorization gives a real
//! speedup. The subtract-add steady state is a serial recurrence —
//! there's nothing to vectorize there.

use super::reductions::{sum, sum_squares};

/// Allocates and returns the rolling mean.
pub fn rolling_mean(values: &[f64], window: usize) -> Vec<f64> {
    let mut out = vec![0.0; values.len()];
    rolling_mean_into(values, window, &mut out);
    out
}

/// Writes the rolling mean into `out`. Panics if lengths disagree or
/// `window == 0`.
pub fn rolling_mean_into(values: &[f64], window: usize, out: &mut [f64]) {
    assert_eq!(
        values.len(),
        out.len(),
        "rolling_mean_into: out.len() must equal values.len()"
    );
    assert!(window >= 1, "rolling_mean_into: window must be >= 1");

    let n = values.len();
    if n == 0 {
        return;
    }

    // Partial-window prefix: values[0..i+1] for i < window-1.
    // This is computed with a SIMD reduction per prefix, but we instead
    // do it with a single running scalar sum since the prefix is short
    // and a SIMD reduction per element would actually be slower.
    let mut running = 0.0_f64;
    let prefix_end = (window - 1).min(n);
    for i in 0..prefix_end {
        running += values[i];
        out[i] = running / (i as f64 + 1.0);
    }

    if n < window {
        return;
    }

    // First full window: values[0..window]. Use SIMD reduction.
    let first_sum = sum(&values[..window]);
    let inv_w = 1.0 / window as f64;
    out[window - 1] = first_sum * inv_w;

    // Steady state: subtract-add running sum. Inherently serial.
    let mut s = first_sum;
    for i in window..n {
        s += values[i] - values[i - window];
        out[i] = s * inv_w;
    }
}

/// Allocates and returns the rolling population standard deviation.
pub fn rolling_std(values: &[f64], window: usize) -> Vec<f64> {
    let mut out = vec![0.0; values.len()];
    rolling_std_into(values, window, &mut out);
    out
}

/// Writes the rolling population standard deviation into `out`.
pub fn rolling_std_into(values: &[f64], window: usize, out: &mut [f64]) {
    assert_eq!(
        values.len(),
        out.len(),
        "rolling_std_into: out.len() must equal values.len()"
    );
    assert!(window >= 1, "rolling_std_into: window must be >= 1");

    let n = values.len();
    if n == 0 {
        return;
    }

    // Partial-window prefix using running sum + sum_sq.
    let mut s = 0.0_f64;
    let mut q = 0.0_f64;
    let prefix_end = (window - 1).min(n);
    for i in 0..prefix_end {
        let v = values[i];
        s += v;
        q += v * v;
        let len = (i + 1) as f64;
        let mean = s / len;
        let var = (q - s * mean) / len;
        out[i] = var.max(0.0).sqrt();
    }

    if n < window {
        return;
    }

    // First full window: SIMD-reduce sum and sum_sq.
    let first_sum = sum(&values[..window]);
    let first_sq = sum_squares(&values[..window]);
    let w_f = window as f64;
    let mean = first_sum / w_f;
    let var = (first_sq - first_sum * mean) / w_f;
    out[window - 1] = var.max(0.0).sqrt();

    // Steady state.
    s = first_sum;
    q = first_sq;
    for i in window..n {
        let inc = values[i];
        let dec = values[i - window];
        s += inc - dec;
        q += inc * inc - dec * dec;
        let mean = s / w_f;
        let var = (q - s * mean) / w_f;
        out[i] = var.max(0.0).sqrt();
    }
}

#[cfg(test)]
pub(crate) fn rolling_mean_scalar(values: &[f64], window: usize) -> Vec<f64> {
    // Mirrors the SIMD subtract-add algorithm; see rolling_std_scalar.
    let n = values.len();
    let mut out = vec![0.0; n];
    if n == 0 {
        return out;
    }

    let mut running = 0.0_f64;
    let prefix_end = (window - 1).min(n);
    for i in 0..prefix_end {
        running += values[i];
        out[i] = running / (i as f64 + 1.0);
    }

    if n < window {
        return out;
    }

    let inv_w = 1.0 / window as f64;
    let mut s = 0.0_f64;
    for &v in &values[..window] {
        s += v;
    }
    out[window - 1] = s * inv_w;

    for i in window..n {
        s += values[i] - values[i - window];
        out[i] = s * inv_w;
    }

    out
}

#[cfg(test)]
pub(crate) fn rolling_std_scalar(values: &[f64], window: usize) -> Vec<f64> {
    // Mirrors the SIMD implementation's subtract-add algorithm so the
    // parity test isolates SIMD-specific divergence (the warm-up sum
    // reduction) from the algorithm's inherent floating-point drift.
    // This is also the algorithm used by the streaming
    // `StandardDeviation` indicator, so SIMD matches streaming.
    let n = values.len();
    let mut out = vec![0.0; n];
    if n == 0 {
        return out;
    }

    let mut s = 0.0_f64;
    let mut q = 0.0_f64;
    let prefix_end = (window - 1).min(n);
    for i in 0..prefix_end {
        let v = values[i];
        s += v;
        q += v * v;
        let len = (i + 1) as f64;
        let mean = s / len;
        let var = (q - s * mean) / len;
        out[i] = var.max(0.0).sqrt();
    }

    if n < window {
        return out;
    }

    let w_f = window as f64;
    s = 0.0;
    q = 0.0;
    for &v in &values[..window] {
        s += v;
        q += v * v;
    }
    let mean = s / w_f;
    let var = (q - s * mean) / w_f;
    out[window - 1] = var.max(0.0).sqrt();

    for i in window..n {
        let inc = values[i];
        let dec = values[i - window];
        s += inc - dec;
        q += inc * inc - dec * dec;
        let mean = s / w_f;
        let var = (q - s * mean) / w_f;
        out[i] = var.max(0.0).sqrt();
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
    fn rolling_mean_known() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        // window=3:
        // i=0: mean(1) = 1
        // i=1: mean(1,2) = 1.5
        // i=2: mean(1,2,3) = 2
        // i=3: mean(2,3,4) = 3
        // i=4: mean(3,4,5) = 4
        let out = rolling_mean(&v, 3);
        let expected = [1.0, 1.5, 2.0, 3.0, 4.0];
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn rolling_mean_parity() {
        for n in [0, 1, 3, 4, 5, 8, 16, 100, 1000, 5000] {
            let values: Vec<f64> = (0..n)
                .map(|i| ((i as f64) * 0.7).sin() * 100.0 + 50.0)
                .collect();
            for w in [1, 2, 3, 4, 5, 8, 32] {
                let s = rolling_mean_scalar(&values, w);
                let v = rolling_mean(&values, w);
                approx_slice_eq(&v, &s, 1e-10, 1e-9);
            }
        }
    }

    #[test]
    fn rolling_std_parity() {
        for n in [0, 1, 3, 4, 5, 8, 16, 100, 1000, 5000] {
            let values: Vec<f64> = (0..n)
                .map(|i| ((i as f64) * 0.31).cos() * 20.0 + 100.0)
                .collect();
            for w in [1, 2, 3, 4, 5, 8, 32] {
                let s = rolling_std_scalar(&values, w);
                let v = rolling_std(&values, w);
                approx_slice_eq(&v, &s, 1e-8, 1e-8);
            }
        }
    }

    #[test]
    fn rolling_std_window_one_is_zero() {
        let v = [5.0, 9.0, 1.0, 100.0];
        let out = rolling_std(&v, 1);
        for x in out {
            assert!(x.abs() < 1e-12);
        }
    }

    #[test]
    #[should_panic(expected = "window must be >= 1")]
    fn rolling_mean_window_zero_panics() {
        let v = [1.0, 2.0];
        let mut out = [0.0; 2];
        rolling_mean_into(&v, 0, &mut out);
    }
}
