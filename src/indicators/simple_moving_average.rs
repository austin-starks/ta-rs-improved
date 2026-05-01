use std::collections::VecDeque;
use std::fmt;
use std::time::Duration;

use crate::indicators::AdaptiveTimeDetector;
use crate::{errors::Result, Next, NextBatch, Reset};
use chrono::{DateTime, Utc};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

const MAX_WINDOW_SIZE: usize = 500;
const KEEP_OLDEST: usize = 10;
const KEEP_RECENT: usize = 100;

#[doc(alias = "SMA")]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct SimpleMovingAverage {
    duration: Duration,
    window: VecDeque<(DateTime<Utc>, f64)>,
    sum: f64,
    detector: AdaptiveTimeDetector,
}

impl SimpleMovingAverage {
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
            detector: AdaptiveTimeDetector::new(duration),
        })
    }

    pub fn get_internal_state(&self) -> (Duration, VecDeque<(DateTime<Utc>, f64)>, f64) {
        (self.duration, self.window.clone(), self.sum)
    }

    fn remove_old_data(&mut self, current_time: DateTime<Utc>) {
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

        // Keep oldest entries
        for i in 0..middle_start.min(len) {
            let entry = self.window[i];
            new_sum += entry.1;
            new_window.push_back(entry);
        }

        // Thin middle - keep every other entry
        let mut keep = true;
        for i in middle_start..middle_end {
            if keep {
                let entry = self.window[i];
                new_sum += entry.1;
                new_window.push_back(entry);
            }
            keep = !keep;
        }

        // Keep newest entries
        for i in middle_end..len {
            let entry = self.window[i];
            new_sum += entry.1;
            new_window.push_back(entry);
        }

        self.window = new_window;
        self.sum = new_sum;
    }
}

impl Next<f64> for SimpleMovingAverage {
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

        // Add new data point
        self.window.push_back((timestamp, value));
        self.sum += value;

        // Thin window if it exceeds max size (sparse sampling for memory efficiency)
        self.thin_window();

        // Calculate moving average
        if self.window.is_empty() {
            0.0
        } else {
            self.sum / self.window.len() as f64
        }
    }
}

impl NextBatch<f64> for SimpleMovingAverage {}

impl Reset for SimpleMovingAverage {
    fn reset(&mut self) {
        self.window.clear();
        self.sum = 0.0;
        self.detector.reset();
    }
}

impl Default for SimpleMovingAverage {
    fn default() -> Self {
        // Use std::time::Duration constructor
        Self::new(Duration::from_secs(14 * 24 * 60 * 60)).unwrap() // 14 days in seconds
    }
}

impl fmt::Display for SimpleMovingAverage {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Use as_secs() instead of Debug format
        write!(f, "SMA({}s)", self.duration.as_secs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    #[test]
    fn test_new() {
        assert!(SimpleMovingAverage::new(Duration::from_secs(0)).is_err());
        assert!(SimpleMovingAverage::new(Duration::from_secs(1)).is_ok());
    }

    #[test]
    fn test_next() {
        let duration = Duration::from_secs(4);
        let mut sma = SimpleMovingAverage::new(duration).unwrap();
        let start_time = Utc::now();
        let elapsed_time = chrono::Duration::seconds(1);
        assert_eq!(sma.next((start_time, 4.0)), 4.0);
        assert_eq!(sma.next((start_time + elapsed_time, 5.0)), 4.5);
        assert_eq!(sma.next((start_time + elapsed_time * 2, 6.0)), 5.0);
        assert_eq!(sma.next((start_time + elapsed_time * 3, 6.0)), 5.25);
        assert_eq!(sma.next((start_time + elapsed_time * 4, 6.0)), 5.75);
        assert_eq!(sma.next((start_time + elapsed_time * 5, 6.0)), 6.0);
        assert_eq!(sma.next((start_time + elapsed_time * 6, 2.0)), 5.0);
        // test explicit out of bounds
        let chrono_duration = chrono::Duration::from_std(duration).unwrap();
        assert_eq!(
            sma.next((start_time + elapsed_time * 6 + chrono_duration, 2.0)),
            2.0
        );
    }

    #[test]
    fn test_reset() {
        let duration = Duration::from_secs(4);
        let mut sma = SimpleMovingAverage::new(duration).unwrap();
        let start_time = Utc::now();
        let elapsed_time = chrono::Duration::seconds(1);
        assert_eq!(sma.next((start_time, 4.0)), 4.0);
        assert_eq!(sma.next((start_time + elapsed_time, 5.0)), 4.5);
        assert_eq!(sma.next((start_time + elapsed_time * 2, 6.0)), 5.0);

        sma.reset();
        assert_eq!(sma.next((start_time + elapsed_time * 3, 99.0)), 99.0);
    }

    #[test]
    fn test_default() {
        let _sma = SimpleMovingAverage::default();
    }

    #[test]
    fn test_display() {
        let indicator = SimpleMovingAverage::new(Duration::from_secs(7)).unwrap();
        assert_eq!(format!("{}", indicator), "SMA(7s)");
    }
}
