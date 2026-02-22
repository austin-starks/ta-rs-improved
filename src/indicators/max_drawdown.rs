use crate::errors::{Result, TaError};
use crate::indicators::AdaptiveTimeDetector;
use crate::{Next, Reset};
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
            })
        }
    }

    fn calculate_max_drawdown(&self) -> f64 {
        // No changes needed here
        let mut peak = f64::MIN;
        let mut max_drawdown = 0.0;
        for &(_, value) in &self.window {
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

    fn remove_old_data(&mut self, current_time: DateTime<Utc>) {
        // Change: Convert std::time::Duration to chrono::Duration for the subtraction
        let chrono_duration = chrono::Duration::from_std(self.duration).unwrap();
        while self
            .window
            .front()
            .map_or(false, |(time, _)| *time < current_time - chrono_duration)
        {
            self.window.pop_front();
        }
    }
}

impl Next<f64> for MaxDrawdown {
    type Output = f64;

    fn next(&mut self, (timestamp, value): (DateTime<Utc>, f64)) -> Self::Output {
        // Check if we should replace the last value (same time bucket)
        let should_replace = self.detector.should_replace(timestamp);

        // ALWAYS remove old data first, regardless of replace/add
        self.remove_old_data(timestamp);

        if should_replace && !self.window.is_empty() {
            self.window.pop_back();
        }
        self.window.push_back((timestamp, value));
        self.calculate_max_drawdown()
    }
}

impl Reset for MaxDrawdown {
    fn reset(&mut self) {
        // No changes needed
        self.window.clear();
        self.detector.reset();
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
}
