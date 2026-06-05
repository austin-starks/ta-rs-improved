use crate::errors::{Result, TaError};
use crate::indicators::window_aggregate::{DrawdownAgg, MonoidWindow, WindowAggregate};
use crate::indicators::AdaptiveTimeDetector;
use crate::{Next, NextBatch, Reset};
use chrono::{DateTime, Utc}; // Remove Duration from here
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt;
use std::time::Duration; // Change: use std::time::Duration

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct MaxDrawdown {
    duration: Duration, // Now std::time::Duration
    window: VecDeque<(DateTime<Utc>, f64)>,
    detector: AdaptiveTimeDetector,
    /// Cached `chrono::Duration` form of `duration` (computed once on first use)
    /// so `next()` skips a `from_std` conversion every call. Not serialized;
    /// lazily recomputed after deserialization.
    #[cfg_attr(feature = "serde", serde(skip))]
    cached_window: Option<i64>,
    // Incremental aggregate over all-but-last window element. Derived state,
    // rebuilt lazily from `window` (e.g. after deserialize) via `ensure_built`.
    #[cfg_attr(feature = "serde", serde(skip))]
    swag: MonoidWindow<DrawdownAgg>,
    #[cfg_attr(feature = "serde", serde(skip))]
    swag_built: bool,
}

impl MaxDrawdown {
    pub fn get_window(&self) -> VecDeque<(DateTime<Utc>, f64)> {
        self.window.clone()
    }

    pub fn new(duration: Duration) -> Result<Self> {
        // Change: std::time::Duration can't be negative, so just check if it's zero
        if duration.as_secs() == 0 && duration.subsec_nanos() == 0 {
            Err(TaError::InvalidParameter)
        } else {
            Ok(Self {
                duration,
                window: VecDeque::new(),
                detector: AdaptiveTimeDetector::new(duration),
                cached_window: None,
                swag: MonoidWindow::default(),
                swag_built: false,
            })
        }
    }

    /// Rebuild `swag` (the aggregate over all-but-last window elements) from
    /// `window`. Needed after deserialize, where `swag` defaults to empty.
    fn ensure_built(&mut self) {
        if self.swag_built {
            return;
        }
        self.swag.clear();
        let committed = self.window.len().saturating_sub(1);
        for &(_, value) in self.window.iter().take(committed) {
            self.swag.push_back(value);
        }
        self.swag_built = true;
    }

    /// Evict expired front elements from both `window` and `swag`. The last
    /// window element is the uncommitted tail (not in `swag`), so only mirror
    /// the pop into `swag` while more than one element remains.
    fn remove_old_data(&mut self, current_time: DateTime<Utc>) {
        let dur_nanos = *self
            .cached_window
            .get_or_insert_with(|| self.duration.as_nanos() as i64);
        let cutoff_nanos = current_time.timestamp_nanos_opt().unwrap_or(i64::MIN) - dur_nanos;
        while self
            .window
            .front()
            .map_or(false, |(time, _)| time.timestamp_nanos_opt().unwrap_or(i64::MIN) < cutoff_nanos)
        {
            if self.window.len() > 1 {
                self.swag.pop_front();
            }
            self.window.pop_front();
        }
    }

    fn current_max_drawdown(&self, tail: f64) -> f64 {
        let agg = match self.swag.aggregate() {
            Some(prefix) => prefix.combine(&DrawdownAgg::leaf(tail)),
            None => DrawdownAgg::leaf(tail),
        };
        100.0 * agg.ratio()
    }
}

impl Next<f64> for MaxDrawdown {
    type Output = f64;

    fn next(&mut self, (timestamp, value): (DateTime<Utc>, f64)) -> Self::Output {
        self.ensure_built();

        // Check if we should replace the last value (same time bucket)
        let should_replace = self.detector.should_replace(timestamp);

        // ALWAYS remove old data first, regardless of replace/add
        self.remove_old_data(timestamp);

        if should_replace && !self.window.is_empty() {
            // Replace the tail: window[..len-1] (and thus `swag`) is untouched.
            self.window.pop_back();
        } else if let Some(&(_, prev_tail)) = self.window.back() {
            // Genuine append: the previous tail becomes a committed element.
            self.swag.push_back(prev_tail);
        }
        self.window.push_back((timestamp, value));
        self.current_max_drawdown(value)
    }
}

impl NextBatch<f64> for MaxDrawdown {}

impl Reset for MaxDrawdown {
    fn reset(&mut self) {
        self.window.clear();
        self.detector.reset();
        self.swag.clear();
        self.swag_built = true;
    }
}

impl fmt::Display for MaxDrawdown {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Change: Use as_secs() instead of num_seconds()
        write!(f, "MaxDrawdown({}s)", self.duration.as_secs())
    }
}

impl Default for MaxDrawdown {
    fn default() -> Self {
        // Change: Use std::time::Duration constructor
        Self::new(Duration::from_secs(14 * 24 * 60 * 60)).unwrap() // 14 days in seconds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_new() {
        assert!(MaxDrawdown::new(Duration::from_secs(0)).is_err());
        assert!(MaxDrawdown::new(Duration::from_secs(1)).is_ok());
    }

    #[test]
    fn test_next() {
        let duration = Duration::from_secs(2);
        let mut max = MaxDrawdown::new(duration).unwrap();
        let start_time = Utc.ymd(2020, 1, 1).and_hms(0, 0, 0);

        // Change: Use chrono::Duration for adding to DateTime
        assert_eq!(max.next((start_time, 4.0)), 0.0);
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(1), 2.0)),
            50.0
        );
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(2), 1.0)),
            75.0
        );
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(3), 3.0)),
            50.0
        );
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(4), 4.0)),
            0.0
        );
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(5), 0.0)),
            100.0
        );
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(6), 2.0)),
            100.0
        );
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(7), 3.0)),
            0.0
        );
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(8), 1.5)),
            50.0
        );
    }

    #[test]
    fn test_reset() {
        let duration = Duration::from_secs(100);
        let mut max = MaxDrawdown::new(duration).unwrap();
        let start_time = Utc.ymd(2020, 1, 1).and_hms(0, 0, 0);

        // Change: Use chrono::Duration for adding to DateTime
        assert_eq!(max.next((start_time, 4.0)), 0.0);
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(50), 10.0)),
            0.0
        );
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(100), 2.0)),
            80.0
        );
        max.reset();
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(150), 4.0)),
            0.0
        );
    }

    #[test]
    fn test_display() {
        let indicator = MaxDrawdown::new(Duration::from_secs(7)).unwrap();
        assert_eq!(format!("{}", indicator), "MaxDrawdown(7s)");
    }

    // Naive O(W) reference: the original full-window rescan.
    fn naive(window: &VecDeque<(DateTime<Utc>, f64)>) -> f64 {
        let mut peak = f64::MIN;
        let mut max_drawdown = 0.0;
        for &(_, value) in window {
            if value > peak {
                peak = value;
            }
            let drawdown = (peak - value) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }
        100.0 * max_drawdown
    }

    #[test]
    fn equivalence_with_naive_under_appends_replaces_and_evictions() {
        let mut state: u64 = 0x9e3779b97f4a7c15;
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let start_time = Utc.ymd(2021, 6, 1).and_hms(0, 0, 0);

        for _ in 0..300 {
            let duration = Duration::from_secs(1 + rng() % 30);
            let mut indicator = MaxDrawdown::new(duration).unwrap();
            let mut offset_ms: i64 = 0;
            for _ in 0..400 {
                // Mix sub-second steps (same bucket -> replace) with larger jumps (eviction).
                offset_ms += (rng() % 1500) as i64;
                let ts = start_time + chrono::Duration::milliseconds(offset_ms);
                // Strictly positive values across a wide range.
                let value = 1.0 + (rng() % 1_000_000) as f64 / 1000.0;
                let got = indicator.next((ts, value));
                let expected = naive(&indicator.get_window());
                assert_eq!(
                    got, expected,
                    "incremental {got} != naive {expected} (dur={duration:?})"
                );
            }
        }
    }
}
