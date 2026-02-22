use chrono::{DateTime, Utc};
use std::collections::VecDeque;
use std::fmt;
use std::time::Duration;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::errors::{Result, TaError};
use crate::indicators::AdaptiveTimeDetector;
use crate::{Next, Reset};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct MeanAbsoluteDeviation {
    duration: Duration, // Now std::time::Duration
    sum: f64,
    window: VecDeque<(DateTime<Utc>, f64)>,
    detector: AdaptiveTimeDetector,
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
            })
        }
    }

    fn remove_old_data(&mut self, current_time: DateTime<Utc>) {
        // Convert std::time::Duration to chrono::Duration for the subtraction
        let chrono_duration = chrono::Duration::from_std(self.duration).unwrap();
        while self
            .window
            .front()
            .map_or(false, |(time, _)| *time <= current_time - chrono_duration)
        {
            if let Some((_, value)) = self.window.pop_front() {
                self.sum -= value;
            }
        }
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
