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
pub struct MaxDrawup {
    duration: Duration, // Now std::time::Duration
    window: VecDeque<(DateTime<Utc>, f64)>,
    detector: AdaptiveTimeDetector,
}

impl MaxDrawup {
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

    fn calculate_max_drawup(&self) -> f64 {
        let mut trough = f64::MAX;
        let mut max_drawup = 0.0;

        for &(_, value) in &self.window {
            if value < trough {
                trough = value;
            }
            let drawup = (value - trough) / trough;
            if drawup > max_drawup {
                max_drawup = drawup;
            }
        }

        100.0 * max_drawup
    }

    fn remove_old_data(&mut self, current_time: DateTime<Utc>) {
        // Change: Convert std::time::Duration to chrono::Duration for date arithmetic
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

impl Next<f64> for MaxDrawup {
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

        // Calculate the maximum drawup within the current window
        self.calculate_max_drawup()
    }
}

impl Reset for MaxDrawup {
    fn reset(&mut self) {
        self.window.clear();
        self.detector.reset();
    }
}

impl fmt::Display for MaxDrawup {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Change: Use as_secs() instead of num_seconds()
        write!(f, "MaxDrawup({}s)", self.duration.as_secs())
    }
}

impl Default for MaxDrawup {
    fn default() -> Self {
        // Change: Use Duration::from_secs for 14 days
        Self::new(Duration::from_secs(14 * 24 * 60 * 60)).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_new() {
        // Change: Use std::time::Duration constructors
        assert!(MaxDrawup::new(Duration::from_secs(0)).is_err());
        assert!(MaxDrawup::new(Duration::from_secs(1)).is_ok());
    }

    #[test]
    fn test_next() {
        let duration = Duration::from_secs(2);
        let mut max = MaxDrawup::new(duration).unwrap();
        let start_time = Utc.ymd(2020, 1, 1).and_hms(0, 0, 0);

        // Use chrono::Duration for date arithmetic
        assert_eq!(max.next((start_time, 4.0)), 0.0);
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(1), 2.0)),
            0.0
        );
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(2), 1.0)),
            0.0
        );
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(3), 3.0)),
            200.0
        );
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(4), 4.0)),
            300.0
        );
        assert_eq!(
            crate::test_helper::round(max.next((start_time + chrono::Duration::seconds(5), 3.0))),
            33.333
        );
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(6), 6.0)),
            100.0
        );
        assert_eq!(
            max.next((start_time + chrono::Duration::seconds(7), 9.0)),
            200.0
        );
    }

    #[test]
    fn test_reset() {
        let duration = Duration::from_secs(100);
        let mut max_drawup = MaxDrawup::new(duration).unwrap();
        let start_time = Utc.ymd(2020, 1, 1).and_hms(0, 0, 0);

        assert_eq!(max_drawup.next((start_time, 4.0)), 0.0);

        assert_eq!(
            max_drawup.next((start_time + chrono::Duration::seconds(50), 10.0)),
            150.0
        );

        assert_eq!(
            max_drawup.next((start_time + chrono::Duration::seconds(100), 2.0)),
            150.0
        );

        max_drawup.reset();

        assert_eq!(
            max_drawup.next((start_time + chrono::Duration::seconds(150), 4.0)),
            0.0
        );

        assert_eq!(
            max_drawup.next((start_time + chrono::Duration::seconds(200), 8.0)),
            100.0
        );
    }

    #[test]
    fn test_display() {
        let indicator = MaxDrawup::new(Duration::from_secs(7)).unwrap();
        assert_eq!(format!("{}", indicator), "MaxDrawup(7s)");
    }
}
