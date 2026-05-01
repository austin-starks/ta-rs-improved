//! SIMD-accelerated batch operations over `&[f64]` slices.
//!
//! These functions complement the streaming `Next`-trait indicators with
//! vectorized paths for bulk array workloads (backtests, historical
//! analysis, vectorized recomputation). They use [`wide`] under the hood,
//! which compiles to AVX2 on modern x86_64, paired NEON on aarch64, and
//! a scalar fallback otherwise — correct on every target, fast where the
//! hardware supports it.
//!
//! Every SIMD function in this module has a `_scalar` reference
//! implementation used for parity testing. The two must agree to within
//! a small floating-point epsilon for all inputs; any divergence is a bug.

pub mod ema;
pub mod rate_of_change;
pub mod reductions;
pub mod rolling;

pub use ema::{ema, ema_continuation_into, ema_into};
pub use rate_of_change::{rate_of_change, rate_of_change_into};
pub use reductions::{mean, std_dev, sum, sum_squares, variance};
pub use rolling::{rolling_mean, rolling_mean_into, rolling_std, rolling_std_into};
