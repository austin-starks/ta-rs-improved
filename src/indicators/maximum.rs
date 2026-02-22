use std::collections::VecDeque;
use std::fmt;
use std::time::Duration; // Change: Use std::time::Duration

use crate::errors::{Result, TaError};
use crate::indicators::AdaptiveTimeDetector;
use crate::{Next, Reset};
use chrono::{DateTime, Utc};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Maximum {
    duration: Duration, // Now std::time::Duration
    window: VecDeque<(DateTime<Utc>, f64)>,
    detector: AdaptiveTimeDetector,
}

impl Maximum {
    pub fn get_window(&self) -> VecDeque<(DateTime<Utc>, f64)> {
        self.window.clone()
    }

    pub fn new(duration: Duration) -> Result<Self> {
        // Change: Check for zero duration (std::time::Duration can't be negative)
        if duration.as_secs() == 0 && duration.subsec_nanos() == 0 {
            Err(TaError::InvalidParameter)
        } else {
            Ok(Self {
                duration,
                window: VecDeque::new(),
                detector: AdaptiveTimeDetector::new(duration),
            })
        }
    }

    fn find_max_value(&self) -> f64 {
        self.window
            .iter()
            .map(|&(_, val)| val)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    fn remove_old_data(&mut self, current_time: DateTime<Utc>) {
        // Change: Convert std::time::Duration to chrono::Duration for date arithmetic
        let chrono_duration = chrono::Duration::from_std(self.duration).unwrap();
        while self
            .window
            .front()
            .map_or(false, |(time, _)| *time <= current_time - chrono_duration)
        {
            self.window.pop_front();
        }
    }
}

impl Default for Maximum {
    fn default() -> Self {
        // Change: Use Duration::from_secs for 14 days
        Self::new(Duration::from_secs(14 * 24 * 60 * 60)).unwrap()
    }
}

impl Next<f64> for Maximum {
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

        // Find the maximum value in the current window
        self.find_max_value()
    }
}

impl Reset for Maximum {
    fn reset(&mut self) {
        self.window.clear();
        self.detector.reset();
    }
}

impl fmt::Display for Maximum {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Change: Use as_secs() instead of num_seconds()
        write!(f, "MAX({}s)", self.duration.as_secs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_new() {
        // Change: Use std::time::Duration constructors
        assert!(Maximum::new(Duration::from_secs(0)).is_err());
        assert!(Maximum::new(Duration::from_secs(1)).is_ok());
    }

    #[test]
    fn test_next() {
        let duration = Duration::from_secs(2);
        let mut max = Maximum::new(duration).unwrap();
        let start_time = Utc.ymd(2020, 1, 1).and_hms(0, 0, 0);

        // Use chrono::Duration for date arithmetic
        assert_eq!(max.next((start_time, 4.0)), 4.0);
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(1), 1.2)),
            4.0
        );
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(2), 5.0)),
            5.0
        );
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(3), 3.0)),
            5.0
        );
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(4), 4.0)),
            4.0
        );
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(5), 0.0)),
            4.0
        );
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(6), -1.0)),
            0.0
        );
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(7), -2.0)),
            -1.0
        );
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(8), -1.5)),
            -1.5
        );
    }

    #[test]
    fn test_reset() {
        let duration = Duration::from_secs(100);
        let mut max = Maximum::new(duration).unwrap();
        let start_time = Utc.ymd(2020, 1, 1).and_hms(0, 0, 0);

        assert_eq!(max.next((start_time, 4.0)), 4.0);
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(50), 10.0)),
            10.0
        );
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(100), 4.0)),
            10.0
        );

        max.reset();
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(150), 4.0)),
            4.0
        );
    }

    #[test]
    fn test_default() {
        let _ = Maximum::default();
    }

    #[test]
    fn test_display() {
        let indicator = Maximum::new(Duration::from_secs(7)).unwrap();
        assert_eq!(format!("{}", indicator), "MAX(7s)");
    }
}
