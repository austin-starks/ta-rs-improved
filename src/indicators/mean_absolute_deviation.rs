use chrono::{DateTime, Utc};
use std::collections::VecDeque;
use std::fmt;
use std::time::Duration;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::errors::{Result, TaError};
use crate::indicators::AdaptiveTimeDetector;
use crate::{Next, NextBatch, Reset};

const MAX_WINDOW_SIZE: usize = 500;
const KEEP_OLDEST: usize = 10;
const KEEP_RECENT: usize = 100;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct MeanAbsoluteDeviation {
    duration: Duration,
    sum: f64,
    window: VecDeque<(DateTime<Utc>, f64)>,
    detector: AdaptiveTimeDetector,
    /// Cached `chrono::Duration` form of `duration` (computed once on first use)
    /// so `next()` skips a `from_std` conversion every call. Not serialized;
    /// lazily recomputed after deserialization.
    #[cfg_attr(feature = "serde", serde(skip))]
    cached_window: Option<chrono::Duration>,
}

impl MeanAbsoluteDeviation {
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
                sum: 0.0,
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
        while self
            .window
            .front()
            .map_or(false, |(time, _)| *time <= cutoff)
        {
            if let Some((_, value)) = self.window.pop_front() {
                self.sum -= value;
            }
        }
    }

    fn thin_window(&mut self) {
        if self.window.len() <= MAX_WINDOW_SIZE {
            return;
        }

        let len = self.window.len();
        let middle_start = KEEP_OLDEST;
        let middle_end = len.saturating_sub(KEEP_RECENT);

        if middle_end <= middle_start {
            return;
        }

        let mut new_window = VecDeque::with_capacity(MAX_WINDOW_SIZE);
        let mut new_sum = 0.0;

        for i in 0..middle_start.min(len) {
            let (ts, val) = self.window[i];
            new_sum += val;
            new_window.push_back((ts, val));
        }

        let mut keep = true;
        for i in middle_start..middle_end {
            if keep {
                let (ts, val) = self.window[i];
                new_sum += val;
                new_window.push_back((ts, val));
            }
            keep = !keep;
        }

        for i in middle_end..len {
            let (ts, val) = self.window[i];
            new_sum += val;
            new_window.push_back((ts, val));
        }

        self.window = new_window;
        self.sum = new_sum;
    }
}

impl Next<f64> for MeanAbsoluteDeviation {
    type Output = f64;

    fn next(&mut self, (timestamp, value): (DateTime<Utc>, f64)) -> Self::Output {
        // Check if we should replace the last value (same time bucket)
        let should_replace = self.detector.should_replace(timestamp);

        // ALWAYS remove old data first, regardless of replace/add
        self.remove_old_data(timestamp);

        if should_replace && !self.window.is_empty() {
            // Replace the last value in the same time bucket
            if let Some((_, old_value)) = self.window.pop_back() {
                self.sum -= old_value;
            }
        }

        self.window.push_back((timestamp, value));
        self.sum += value;

        // Thin window if it exceeds max size (sparse sampling for memory efficiency)
        self.thin_window();

        let mean = self.sum / self.window.len() as f64;

        let mut mad = 0.0;
        for &(_, val) in &self.window {
            mad += (val - mean).abs();
        }

        if self.window.is_empty() {
            0.0
        } else {
            mad / self.window.len() as f64
        }
    }
}

impl NextBatch<f64> for MeanAbsoluteDeviation {}

impl Reset for MeanAbsoluteDeviation {
    fn reset(&mut self) {
        self.sum = 0.0;
        self.window.clear();
        self.detector.reset();
    }
}

impl Default for MeanAbsoluteDeviation {
    fn default() -> Self {
        // Use std::time::Duration constructor
        Self::new(Duration::from_secs(14 * 24 * 60 * 60)).unwrap() // 14 days in seconds
    }
}

impl fmt::Display for MeanAbsoluteDeviation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Use as_secs() instead of Debug format
        write!(f, "MAD({}s)", self.duration.as_secs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helper::*;
    use chrono::{TimeZone, Utc};

    // Helper function to create a Utc DateTime from a timestamp
    fn to_utc_datetime(timestamp: i64) -> DateTime<Utc> {
        Utc.timestamp(timestamp, 0)
    }

    #[test]
    fn test_new() {
        assert!(MeanAbsoluteDeviation::new(Duration::from_secs(0)).is_err());
        assert!(MeanAbsoluteDeviation::new(Duration::from_secs(1)).is_ok());
    }

    #[test]
    fn test_next() {
        let duration = Duration::from_secs(5);
        let mut mad = MeanAbsoluteDeviation::new(duration).unwrap();

        let timestamp1 = to_utc_datetime(0);
        let timestamp2 = to_utc_datetime(1);
        let timestamp3 = to_utc_datetime(2);
        let timestamp4 = to_utc_datetime(3);
        let timestamp5 = to_utc_datetime(4);
        let timestamp6 = to_utc_datetime(5);

        assert_eq!(round(mad.next((timestamp1, 1.5))), 0.0);
        assert_eq!(round(mad.next((timestamp2, 4.0))), 1.25);
        assert_eq!(round(mad.next((timestamp3, 8.0))), 2.333);
        assert_eq!(round(mad.next((timestamp4, 4.0))), 1.813);
        assert_eq!(round(mad.next((timestamp5, 4.0))), 1.48);
        assert_eq!(round(mad.next((timestamp6, 1.5))), 1.48);
    }

    #[test]
    fn test_reset() {
        let duration = Duration::from_secs(5);
        let mut mad = MeanAbsoluteDeviation::new(duration).unwrap();

        let timestamp1 = to_utc_datetime(0);
        let timestamp2 = to_utc_datetime(1);

        assert_eq!(round(mad.next((timestamp1, 1.5))), 0.0);
        assert_eq!(round(mad.next((timestamp2, 4.0))), 1.25);

        mad.reset();

        assert_eq!(round(mad.next((timestamp1, 1.5))), 0.0);
        assert_eq!(round(mad.next((timestamp2, 4.0))), 1.25);
    }

    #[test]
    fn test_default() {
        MeanAbsoluteDeviation::default();
    }

    #[test]
    fn test_display() {
        let indicator = MeanAbsoluteDeviation::new(Duration::from_secs(10)).unwrap();
        assert_eq!(format!("{}", indicator), "MAD(10s)");
    }
}
