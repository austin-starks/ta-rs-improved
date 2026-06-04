use std::collections::VecDeque;
use std::fmt;
use std::time::Duration;

use crate::errors::{Result, TaError};
use crate::indicators::AdaptiveTimeDetector;
use crate::traits::{Next, NextBatch, Reset};
use chrono::{DateTime, Utc};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

const MAX_WINDOW_SIZE: usize = 500;
const KEEP_OLDEST: usize = 10;
const KEEP_RECENT: usize = 100;

#[doc(alias = "ROC")]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct RateOfChange {
    duration: Duration,
    window: VecDeque<(DateTime<Utc>, f64)>,
    detector: AdaptiveTimeDetector,
    /// Cached `chrono::Duration` form of `duration` (computed once on first use)
    /// so `next()` skips a `from_std` conversion every call. Not serialized;
    /// lazily recomputed after deserialization.
    #[cfg_attr(feature = "serde", serde(skip))]
    cached_window: Option<chrono::Duration>,
}

impl RateOfChange {
    pub fn get_window(&self) -> VecDeque<(DateTime<Utc>, f64)> {
        self.window.clone()
    }
    pub fn new(duration: Duration) -> Result<Self> {
        // std::time::Duration can't be negative, so just check if it's zero
        if duration.as_secs() == 0 && duration.subsec_nanos() == 0 {
            Err(TaError::InvalidParameter)
        } else {
            Ok(Self {
                duration,
                window: VecDeque::new(),
                detector: AdaptiveTimeDetector::new(duration),
                cached_window: None,
            })
        }
    }

    fn remove_old_data(&mut self, current_time: DateTime<Utc>) {
        let chrono_duration = *self
            .cached_window
            .get_or_insert_with(|| chrono::Duration::from_std(self.duration).unwrap());
        let cutoff = current_time - chrono_duration;
        while self.window.front().map_or(false, |(time, _)| *time < cutoff) {
            self.window.pop_front();
        }
    }

    fn thin_window(&mut self) {
        if self.window.len() <= MAX_WINDOW_SIZE {
            return;
        }

        // Keep oldest KEEP_OLDEST entries, thin middle, keep newest KEEP_RECENT entries
        let len = self.window.len();
        let middle_start = KEEP_OLDEST;
        let middle_end = len.saturating_sub(KEEP_RECENT);

        if middle_end <= middle_start {
            return;
        }

        // Remove every other entry from the middle section
        let mut new_window = VecDeque::with_capacity(MAX_WINDOW_SIZE);

        // Keep oldest entries
        for i in 0..middle_start.min(len) {
            new_window.push_back(self.window[i]);
        }

        // Thin middle - keep every other entry
        let mut keep = true;
        for i in middle_start..middle_end {
            if keep {
                new_window.push_back(self.window[i]);
            }
            keep = !keep;
        }

        // Keep newest entries
        for i in middle_end..len {
            new_window.push_back(self.window[i]);
        }

        self.window = new_window;
    }
}

impl Next<f64> for RateOfChange {
    type Output = f64;

    fn next(&mut self, (timestamp, value): (DateTime<Utc>, f64)) -> Self::Output {
        // Check if we should replace the last value (same time bucket)
        let should_replace = self.detector.should_replace(timestamp);

        // ALWAYS remove old data first, regardless of replace/add
        self.remove_old_data(timestamp);

        if should_replace && !self.window.is_empty() {
            // Replace the last value in the same time bucket
            self.window.pop_back();
        }

        // Add the new data point
        self.window.push_back((timestamp, value));

        // Thin window if it exceeds max size (sparse sampling for memory efficiency)
        self.thin_window();

        // Calculate the rate of change if we have at least two data points
        if self.window.len() > 1 {
            let (oldest_time, oldest_value) =
                self.window.front().expect("Window has at least one item");
            let (newest_time, newest_value) =
                self.window.back().expect("Window has at least one item");

            // Ensure we do not divide by zero
            if oldest_value.clone() != 0.0 {
                (newest_value - oldest_value) / oldest_value * 100.0
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
}

impl NextBatch<f64> for RateOfChange {}

impl Default for RateOfChange {
    fn default() -> Self {
        // Use std::time::Duration constructor
        Self::new(Duration::from_secs(14 * 24 * 60 * 60)).unwrap()  // 14 days in seconds
    }
}

impl fmt::Display for RateOfChange {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Use as_secs() instead of Debug format
        write!(f, "ROC({}s)", self.duration.as_secs())
    }
}

impl Reset for RateOfChange {
    fn reset(&mut self) {
        self.window.clear();
        self.detector.reset();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helper::*;
    use chrono::{TimeZone, Utc};

    test_indicator!(RateOfChange);
    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_new() {
        assert!(RateOfChange::new(Duration::from_secs(0)).is_err());
        assert!(RateOfChange::new(Duration::from_secs(1)).is_ok());
        assert!(RateOfChange::new(Duration::from_secs(100_000)).is_ok());
    }

    #[test]
    fn test_next_f64() {
        let mut roc = RateOfChange::new(Duration::from_secs(3)).unwrap();
        let start_time = Utc.ymd(2020, 1, 1).and_hms(0, 0, 0);

        assert_eq!(round(roc.next((start_time, 10.0))), 0.0);
        assert_eq!(
            round(roc.next((start_time + chrono::Duration::seconds(1), 10.4))),
            4.0
        );
        assert_eq!(
            round(roc.next((start_time + chrono::Duration::seconds(2), 10.57))),
            5.7
        );
        assert_eq!(
            round(roc.next((start_time + chrono::Duration::seconds(3), 10.8))),
            8.0
        );
        assert_eq!(
            round(roc.next((start_time + chrono::Duration::seconds(4), 10.9))),
            4.808
        );
        assert_eq!(
            round(roc.next((start_time + chrono::Duration::seconds(5), 10.0))),
            -5.393
        );
    }

    #[test]
    fn test_reset() {
        let mut roc = RateOfChange::new(Duration::from_secs(3)).unwrap();
        let start_time = Utc.ymd(2020, 1, 1).and_hms(0, 0, 0);

        roc.next((start_time, 12.3));
        roc.next((start_time + chrono::Duration::seconds(1), 15.0));

        roc.reset();

        assert_eq!(round(roc.next((start_time, 10.0))), 0.0);
        assert_eq!(
            round(roc.next((start_time + chrono::Duration::seconds(1), 10.4))),
            4.0
        );
        assert_eq!(
            round(roc.next((start_time + chrono::Duration::seconds(2), 10.57))),
            5.7
        );
    }
}
