use std::collections::VecDeque;
use std::fmt;
use std::time::Duration;

use crate::errors::Result;
use crate::indicators::AdaptiveTimeDetector;
use crate::{Next, Reset};
use chrono::{DateTime, Utc};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[doc(alias = "SD")]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct StandardDeviation {
    duration: Duration, // Now std::time::Duration
    window: VecDeque<(DateTime<Utc>, f64)>,
    sum: f64,
    sum_sq: f64,
    detector: AdaptiveTimeDetector,
}

impl StandardDeviation {
    pub fn get_window(&self) -> VecDeque<(DateTime<Utc>, f64)> {
        self.window.clone()
    }
    pub fn new(duration: Duration) -> Result<Self> {
        // std::time::Duration can't be negative, so just check if it's zero
        if duration.as_secs() == 0 && duration.subsec_nanos() == 0 {
            return Err(crate::errors::TaError::InvalidParameter);
        }
        Ok(Self {
            duration,
            window: VecDeque::new(),
            sum: 0.0,
            sum_sq: 0.0,
            detector: AdaptiveTimeDetector::new(duration),
        })
    }

    // Helper method to remove old data points
    fn remove_old_data(&mut self, current_time: DateTime<Utc>) {
        // Convert std::time::Duration to chrono::Duration for the subtraction
        let chrono_duration = chrono::Duration::from_std(self.duration).unwrap();
        while self
            .window
            .front()
            .map_or(false, |(time, _)| *time <= current_time - chrono_duration)
        {
            if let Some((_, old_value)) = self.window.pop_front() {
                self.sum -= old_value;
                self.sum_sq -= old_value * old_value;
            }
        }
    }

    // Calculate the mean based on the current window
    pub(super) fn mean(&self) -> f64 {
        if !self.window.is_empty() {
            self.sum / self.window.len() as f64
        } else {
            0.0
        }
    }
}

impl Next<f64> for StandardDeviation {
    type Output = f64;
    fn next(&mut self, input: (DateTime<Utc>, f64)) -> Self::Output {
        let (timestamp, value) = input;

        // Check if we should replace the last value (same time bucket)
        let should_replace = self.detector.should_replace(timestamp);

        // ALWAYS remove old data first, regardless of replace/add
        self.remove_old_data(timestamp);

        if should_replace && !self.window.is_empty() {
            // Replace the last value in the same time bucket
            if let Some((_, old_value)) = self.window.pop_back() {
                self.sum -= old_value;
                self.sum_sq -= old_value * old_value;
            }
        }

        // Add new value to the window
        self.window.push_back((timestamp, value));
        self.sum += value;
        self.sum_sq += value * value;

        // Calculate the population standard deviation
        let n = self.window.len() as f64;
        if n == 0.0 {
            0.0
        } else {
            let mean = self.sum / n;
            let variance = (self.sum_sq - (self.sum * mean)) / n;
            variance.sqrt()
        }
    }
}

impl Reset for StandardDeviation {
    fn reset(&mut self) {
        self.window.clear();
        self.sum = 0.0;
        self.sum_sq = 0.0;
        self.detector.reset();
    }
}

impl Default for StandardDeviation {
    fn default() -> Self {
        // Use std::time::Duration constructor
        Self::new(Duration::from_secs(14 * 24 * 60 * 60)).unwrap() // 14 days in seconds
    }
}

impl fmt::Display for StandardDeviation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Use as_secs() instead of Debug format
        write!(f, "SD({}s)", self.duration.as_secs())
    }
}

#[cfg(test)]
mod tests {
    use crate::test_helper::round;

    use super::*;
    use chrono::{TimeZone, Utc};

    #[test]
    fn test_new() {
        assert!(StandardDeviation::new(Duration::from_secs(0)).is_err());
        assert!(StandardDeviation::new(Duration::from_secs(1)).is_ok());
    }

    #[test]
    fn test_next() {
        let duration = Duration::from_secs(4);
        let mut sd = StandardDeviation::new(duration).unwrap();
        let now = Utc::now();
        // Use chrono::Duration for adding to DateTime
        assert_eq!(sd.next((now + chrono::Duration::seconds(1), 10.0)), 0.0);
        assert_eq!(sd.next((now + chrono::Duration::seconds(2), 20.0)), 5.0);
        assert_eq!(
            round(sd.next((now + chrono::Duration::seconds(3), 30.0))),
            8.165
        );
        assert_eq!(
            round(sd.next((now + chrono::Duration::seconds(4), 20.0))),
            7.071
        );
        assert_eq!(
            round(sd.next((now + chrono::Duration::seconds(5), 10.0))),
            7.071
        );
        assert_eq!(
            round(sd.next((now + chrono::Duration::seconds(6), 100.0))),
            35.355
        );
    }

    #[test]
    fn test_reset() {
        let duration = Duration::from_secs(4);
        let mut sd = StandardDeviation::new(duration).unwrap();
        let now = Utc::now();
        assert_eq!(sd.next((now, 10.0)), 0.0);
        assert_eq!(sd.next((now + chrono::Duration::seconds(1), 20.0)), 5.0);
        assert_eq!(
            round(sd.next((now + chrono::Duration::seconds(2), 30.0))),
            8.165
        );

        sd.reset();
        assert_eq!(sd.next((now + chrono::Duration::seconds(3), 20.0)), 0.0);
    }

    #[test]
    fn test_default() {
        let _sd = StandardDeviation::default();
    }

    #[test]
    fn test_display() {
        let indicator = StandardDeviation::new(Duration::from_secs(7)).unwrap();
        assert_eq!(format!("{}", indicator), "SD(7s)");
    }
}
