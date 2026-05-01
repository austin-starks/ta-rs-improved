use std::fmt;
use std::time::Duration;

use crate::errors::Result;
use crate::indicators::{AdaptiveTimeDetector, ExponentialMovingAverage as Ema};
use crate::{Next, NextBatch, Reset};
use chrono::{DateTime, Utc};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[doc(alias = "RSI")]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct RelativeStrengthIndex {
    duration: Duration,
    up_ema_indicator: Ema,
    down_ema_indicator: Ema,
    prev_val: Option<f64>,
    detector: AdaptiveTimeDetector,
}

impl RelativeStrengthIndex {
    pub fn new(duration: Duration) -> Result<Self> {
        Ok(Self {
            duration,
            up_ema_indicator: Ema::new(duration)?,
            down_ema_indicator: Ema::new(duration)?,
            prev_val: None,
            detector: AdaptiveTimeDetector::new(duration),
        })
    }
}

impl Next<f64> for RelativeStrengthIndex {
    type Output = f64;

    fn next(&mut self, (timestamp, value): (DateTime<Utc>, f64)) -> Self::Output {
        // Check if we should replace the last value (same time bucket)
        let should_replace = self.detector.should_replace(timestamp);

        // Calculate gain and loss using the stable prev_val
        let (gain, loss) = if let Some(prev_val) = self.prev_val {
            if value > prev_val {
                (value - prev_val, 0.0)
            } else {
                (0.0, prev_val - value)
            }
        } else {
            (0.0, 0.0)
        };

        // Only update prev_val for the NEXT period if this is not a replacement
        // When replacing, prev_val stays as the previous period's close
        if !should_replace {
            self.prev_val = Some(value);
        }

        // Update EMAs
        let avg_up = self.up_ema_indicator.next((timestamp, gain));
        let avg_down = self.down_ema_indicator.next((timestamp, loss));

        // Calculate and return RSI
        if avg_down == 0.0 {
            if avg_up == 0.0 {
                50.0 // Neutral value when no movement
            } else {
                100.0 // Max value when only gains
            }
        } else {
            let rs = avg_up / avg_down;
            100.0 - (100.0 / (1.0 + rs))
        }
    }
}

impl NextBatch<f64> for RelativeStrengthIndex {
    /// Batched RSI: vectorized gain/loss diff + delegate to
    /// `EMA::next_batch` (SIMD closed-form) on each. Falls back to the
    /// scalar `next` loop if any input would trigger same-bucket
    /// replacement on this RSI's detector — the recurrence form
    /// doesn't handle replacements.
    ///
    /// Output agrees with repeated `next` calls within ~1 ULP per
    /// element (the inner EMA SIMD has ~10 ULP drift; one of those
    /// drifts feeds the gain track, the other feeds the loss track,
    /// and the final RSI ratio multiplies them — so total tolerance
    /// is wider than EMA's. Parity tests use `1e-9` relative).
    fn next_batch(&mut self, inputs: &[(DateTime<Utc>, f64)]) -> Vec<Self::Output> {
        if inputs.is_empty() {
            return Vec::new();
        }

        // Probe the RSI's detector. The two inner EMAs share the same
        // duration → same bucket size → same `should_replace` answer
        // for any timestamp, so probing once is enough.
        let mut probe = self.detector.clone();
        for &(ts, _) in inputs {
            if probe.should_replace(ts) {
                return inputs.iter().map(|&i| self.next(i)).collect();
            }
        }

        // Vectorizable diff loop: compute gains and losses across the
        // batch, threading `prev_val` through.
        let n = inputs.len();
        let mut gain_inputs: Vec<(DateTime<Utc>, f64)> = Vec::with_capacity(n);
        let mut loss_inputs: Vec<(DateTime<Utc>, f64)> = Vec::with_capacity(n);
        let mut prev = self.prev_val;
        for &(ts, value) in inputs {
            let (gain, loss) = match prev {
                Some(p) => {
                    if value > p {
                        (value - p, 0.0)
                    } else {
                        (0.0, p - value)
                    }
                }
                None => (0.0, 0.0),
            };
            gain_inputs.push((ts, gain));
            loss_inputs.push((ts, loss));
            prev = Some(value);
        }

        // Delegate to EMA::next_batch — SIMD closed-form internally.
        let avg_ups = self.up_ema_indicator.next_batch(&gain_inputs);
        let avg_downs = self.down_ema_indicator.next_batch(&loss_inputs);

        // Commit RSI's detector + prev_val. The EMAs already committed
        // their own state inside `next_batch`.
        for &(ts, _) in inputs {
            self.detector.should_replace(ts);
        }
        self.prev_val = Some(inputs[n - 1].1);

        // Combine into the RSI value per index.
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let avg_up = avg_ups[i];
            let avg_down = avg_downs[i];
            let rsi = if avg_down == 0.0 {
                if avg_up == 0.0 {
                    50.0
                } else {
                    100.0
                }
            } else {
                let rs = avg_up / avg_down;
                100.0 - (100.0 / (1.0 + rs))
            };
            out.push(rsi);
        }
        out
    }
}

impl Reset for RelativeStrengthIndex {
    fn reset(&mut self) {
        self.prev_val = None;
        self.up_ema_indicator.reset();
        self.down_ema_indicator.reset();
        self.detector.reset();
    }
}

impl Default for RelativeStrengthIndex {
    fn default() -> Self {
        // Change: Use Duration::from_secs for 14 days
        Self::new(Duration::from_secs(14 * 24 * 60 * 60)).unwrap()
    }
}

impl fmt::Display for RelativeStrengthIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Change: Calculate days from seconds
        let days = self.duration.as_secs() / 86400;
        write!(f, "RSI({} days)", days)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helper::*;
    use chrono::{TimeZone, Utc};

    test_indicator!(RelativeStrengthIndex);

    #[test]
    fn test_new() {
        // Change: Use std::time::Duration constructors
        assert!(RelativeStrengthIndex::new(Duration::from_secs(0)).is_err());
        assert!(RelativeStrengthIndex::new(Duration::from_secs(86400)).is_ok());
        // 1 day
    }

    #[test]
    fn test_next() {
        let mut rsi = RelativeStrengthIndex::new(Duration::from_secs(3 * 86400)).unwrap(); // 3 days
        let timestamp = Utc.ymd(2020, 1, 1).and_hms(0, 0, 0);

        // First value: 10.0 (no previous value, so RSI = 50)
        assert_eq!(rsi.next((timestamp, 10.0)), 50.0);

        // Second value: 10.5 (gain of 0.5, no loss)
        assert_eq!(
            rsi.next((timestamp + chrono::Duration::days(1), 10.5))
                .round(),
            100.0
        );

        // Third value: 10.0 (loss of 0.5 from 10.5)
        // With EMA k=0.5: avg_up=0.125, avg_down=0.25, RS=0.5, RSI=33.33
        assert_eq!(
            rsi.next((timestamp + chrono::Duration::days(2), 10.0))
                .round(),
            33.0
        );

        // Fourth value: 9.5 (loss of 0.5 from 10.0)
        // With continued losses, RSI should drop further
        // avg_up = 0.0625, avg_down = 0.375, RS = 0.1667, RSI = 14.3
        assert_eq!(
            rsi.next((timestamp + chrono::Duration::days(3), 9.5))
                .round(),
            14.0
        );
    }

    #[test]
    fn test_reset() {
        let mut rsi = RelativeStrengthIndex::new(Duration::from_secs(3 * 86400)).unwrap(); // 3 days
        let timestamp = Utc.ymd(2020, 1, 1).and_hms(0, 0, 0);
        assert_eq!(rsi.next((timestamp, 10.0)), 50.0);
        assert_eq!(
            rsi.next((timestamp + chrono::Duration::days(1), 10.5))
                .round(),
            100.0
        );

        rsi.reset();
        assert_eq!(rsi.next((timestamp, 10.0)).round(), 50.0);
        assert_eq!(
            rsi.next((timestamp + chrono::Duration::days(1), 10.5))
                .round(),
            100.0
        );
    }

    #[test]
    fn test_default() {
        RelativeStrengthIndex::default();
    }

    #[test]
    fn test_display() {
        let rsi = RelativeStrengthIndex::new(Duration::from_secs(16 * 86400)).unwrap(); // 16 days
        assert_eq!(format!("{}", rsi), "RSI(16 days)");
    }

    /// SIMD `next_batch` must agree with repeated scalar `next` calls
    /// across many sizes and durations on regular-cadence input. Tolerance
    /// is `1e-9` relative because RSI composes two SIMD EMAs (each ~10 ULP
    /// drift), and the final ratio multiplies them.
    #[test]
    fn test_next_batch_matches_next_loop() {
        for n in [0usize, 1, 2, 4, 5, 16, 17, 100, 1000] {
            for period_days in [1u64, 7, 14, 30, 90] {
                let duration = Duration::from_secs(period_days * 86400);
                let mut a = RelativeStrengthIndex::new(duration).unwrap();
                let mut b = RelativeStrengthIndex::new(duration).unwrap();

                let start = Utc::now();
                let inputs: Vec<(DateTime<Utc>, f64)> = (0..n)
                    .map(|i| {
                        (
                            start + chrono::Duration::days(i as i64),
                            100.0 + ((i as f64) * 0.13).sin() * 10.0,
                        )
                    })
                    .collect();

                let scalar: Vec<f64> = inputs.iter().map(|&i| a.next(i)).collect();
                let batch = b.next_batch(&inputs);

                assert_eq!(scalar.len(), batch.len());
                for (i, (s, v)) in scalar.iter().zip(batch.iter()).enumerate() {
                    let diff = (s - v).abs();
                    let tol = 1e-9 * s.abs().max(v.abs()).max(1.0);
                    assert!(
                        diff <= tol,
                        "n={} period={}d idx={} scalar={} batch={} diff={}",
                        n, period_days, i, s, v, diff
                    );
                }

                // Internal state must agree post-batch — extending with more
                // `next` calls should produce matching output between the
                // two RSI instances.
                let extra: Vec<(DateTime<Utc>, f64)> = (n..n + 5)
                    .map(|i| {
                        (
                            start + chrono::Duration::days(i as i64),
                            42.0 + (i as f64).cos(),
                        )
                    })
                    .collect();
                for &inp in &extra {
                    let s = a.next(inp);
                    let v = b.next(inp);
                    let diff = (s - v).abs();
                    let tol = 1e-9 * s.abs().max(v.abs()).max(1.0);
                    assert!(
                        diff <= tol,
                        "post-batch state diverged: scalar={} batch={}",
                        s, v
                    );
                }
            }
        }
    }

    /// When two timestamps fall in the same intraday bucket, `next_batch`
    /// must fall back to the scalar loop. Output is bit-identical because
    /// the fallback IS the scalar loop.
    #[test]
    fn test_next_batch_falls_back_on_replacement() {
        let duration = Duration::from_secs(60 * 60); // 1-hour RSI
        let mut a = RelativeStrengthIndex::new(duration).unwrap();
        let mut b = RelativeStrengthIndex::new(duration).unwrap();

        let start = Utc::now();
        let inputs = vec![
            (start, 100.0),
            (start + chrono::Duration::minutes(1), 101.0),
            (start + chrono::Duration::minutes(1), 102.0), // same bucket → replace
            (start + chrono::Duration::minutes(2), 103.0),
        ];

        let scalar: Vec<f64> = inputs.iter().map(|&i| a.next(i)).collect();
        let batch = b.next_batch(&inputs);

        for (s, v) in scalar.iter().zip(batch.iter()) {
            assert!(
                (s - v).abs() < 1e-12,
                "scalar={} batch={}",
                s, v
            );
        }
    }
}
