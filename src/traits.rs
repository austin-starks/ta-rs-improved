// Indicator traits

use chrono::{DateTime, Utc};

/// Resets an indicator to the initial state.
pub trait Reset {
    fn reset(&mut self);
}

/// Consumes a data item of type `T` and returns `Output`.
///
/// Typically `T` can be `f64` or a struct similar to [DataItem](struct.DataItem.html), that implements
/// traits necessary to calculate value of a particular indicator.
///
/// In most cases `Output` is `f64`, but sometimes it can be different. For example for
/// [MACD](indicators/struct.MovingAverageConvergenceDivergence.html) it is `(f64, f64, f64)` since
/// MACD returns 3 values.
///
pub trait Next<T> {
    type Output;
    fn next(&mut self, input: (DateTime<Utc>, T)) -> Self::Output;
}

/// Batched form of [`Next`]: consumes a slice of inputs and returns a vector of outputs.
///
/// The default implementation simply loops over [`Next::next`], so any
/// indicator implementing `Next<T>` automatically gets a `next_batch`
/// method with no extra code. Indicators that have a faster vectorized
/// path (e.g. [`ExponentialMovingAverage`](crate::indicators::ExponentialMovingAverage))
/// override this method to dispatch to the SIMD primitives in
/// [`crate::simd`]. Override semantics must exactly match the scalar
/// loop — every overrider has parity tests that compare `next_batch`
/// against repeated calls to `next` on the same input.
pub trait NextBatch<T>: Next<T>
where
    T: Copy,
{
    fn next_batch(&mut self, inputs: &[(DateTime<Utc>, T)]) -> Vec<Self::Output> {
        let mut out = Vec::with_capacity(inputs.len());
        for &input in inputs {
            out.push(self.next(input));
        }
        out
    }
}


