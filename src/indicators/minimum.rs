use std::collections::VecDeque;
use std::fmt;
use std::time::Duration; // Change: Use std::time::Duration

use crate::errors::Result;
use crate::indicators::AdaptiveTimeDetector;
use crate::{Next, Reset};
use chrono::{DateTime, Utc};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Minimum {
    duration: Duration, // Now std::time::Duration
    window: VecDeque<(DateTime<Utc>, f64)>,
    min_value: f64,
    detector: AdaptiveTimeDetector,
}

impl Minimum {
    pub fn get_window(&self) -> VecDeque<(DateTime<Utc>, f64)> {
        self.window.clone()
    }

    pub fn new(duration: Duration) -> Result<Self> {
        // Change: Check for zero duration (std::time::Duration can't be negative)
        if duration.as_secs() == 0 && duration.subsec_nanos() == 0 {
            return Err(crate::errors::TaError::InvalidParameter);
        }
        Ok(Self {
            duration,
            window: VecDeque::new(),
            min_value: f64::INFINITY,
            detector: AdaptiveTimeDetector::new(duration),
        })
    }

    fn update_min(&mut self) {
        self.min_value = self
            .window
            .iter()
            .map(|&(_, val)| val)
            .fold(f64::INFINITY, f64::min);
    }

    fn remove_old(&mut self, current_time: DateTime<Utc>) {
        // Change: Convert std::time::Duration to chrono::Duration for date arithmetic
        let chrono_duration = chrono::Duration::from_std(self.duration).unwrap();
        while self
            .window
            .front()
            .map_or(false, |&(time, _)| time < current_time - chrono_duration)
        {
            self.window.pop_front();
        }
    }
}

impl Next<f64> for Minimum {
    type Output = f64;

    fn next(&mut self, (timestamp, value): (DateTime<Utc>, f64)) -> Self::Output {
        // Check if we should replace the last value (same time bucket)
        let should_replace = self.detector.should_replace(timestamp);

        if should_replace && !self.window.is_empty() {
            // Replace the last value in the same time bucket
            self.window.pop_back();
        } else {
            // New time period - remove old data first
            self.remove_old(timestamp);
        }

        self.window.push_back((timestamp, value));

        if value < self.min_value {
            self.min_value = value;
        } else {
            self.update_min();
        }

        self.min_value
    }
}

impl Reset for Minimum {
    fn reset(&mut self) {
        self.window.clear();
        self.min_value = f64::INFINITY;
        self.detector.reset();
    }
}

impl Default for Minimum {
    fn default() -> Self {
        // Change: Use Duration::from_secs for 14 days
        Self::new(Duration::from_secs(14 * 24 * 60 * 60)).unwrap()
    }
}

impl fmt::Display for Minimum {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Change: Calculate days from seconds
        let days = self.duration.as_secs() / 86400;
        write!(f, "MIN({} days)", days)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    // Helper function to create a DateTime<Utc> from a date string for testing
    fn datetime(s: &str) -> DateTime<Utc> {
        Utc.datetime_from_str(s, "%Y-%m-%d %H:%M:%S").unwrap()
    }

    #[test]
    fn test_new() {
        // Change: Use std::time::Duration constructors
        assert!(Minimum::new(Duration::from_secs(0)).is_err());
        assert!(Minimum::new(Duration::from_secs(86400)).is_ok()); // 1 day
    }

    #[test]
    fn test_next() {
        let duration = Duration::from_secs(2 * 86400); // 2 days
        let mut min = Minimum::new(duration).unwrap();

        assert_eq!(min.next((datetime("2023-01-01 00:00:00"), 4.0)), 4.0);
        assert_eq!(min.next((datetime("2023-01-02 00:00:00"), 1.2)), 1.2);
        assert_eq!(min.next((datetime("2023-01-03 00:00:00"), 5.0)), 1.2);
        assert_eq!(min.next((datetime("2023-01-04 00:00:00"), 3.0)), 1.2);
        assert_eq!(min.next((datetime("2023-01-05 00:00:00"), 4.0)), 3.0);
        assert_eq!(min.next((datetime("2023-01-06 00:00:00"), 6.0)), 3.0);
        assert_eq!(min.next((datetime("2023-01-07 00:00:00"), 7.0)), 4.0);
        assert_eq!(min.next((datetime("2023-01-08 00:00:00"), 8.0)), 6.0);
        assert_eq!(min.next((datetime("2023-01-09 00:00:00"), -9.0)), -9.0);
        assert_eq!(min.next((datetime("2023-01-10 00:00:00"), 0.0)), -9.0);
    }

    #[test]
    fn test_reset() {
        let duration = Duration::from_secs(10 * 86400); // 10 days
        let mut min = Minimum::new(duration).unwrap();

        assert_eq!(min.next((datetime("2023-01-01 00:00:00"), 5.0)), 5.0);
        assert_eq!(min.next((datetime("2023-01-02 00:00:00"), 7.0)), 5.0);

        min.reset();
        assert_eq!(min.next((datetime("2023-01-03 00:00:00"), 8.0)), 8.0);
    }

    #[test]
    fn test_default() {
        let _ = Minimum::default();
    }

    #[test]
    fn test_display() {
        let indicator = Minimum::new(Duration::from_secs(10 * 86400)).unwrap(); // 10 days
        assert_eq!(format!("{}", indicator), "MIN(10 days)");
    }
}
