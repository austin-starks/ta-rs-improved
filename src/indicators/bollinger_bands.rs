use chrono::{DateTime, Utc};
use std::collections::VecDeque;
use std::fmt;
use std::time::Duration; // Change: Use std::time::Duration

use crate::errors::Result;
use crate::indicators::{AdaptiveTimeDetector, StandardDeviation as Sd};
use crate::{Next, Reset};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[doc(alias = "BB")]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct BollingerBands {
    duration: Duration, // Now std::time::Duration
    multiplier: f64,
    sd: Sd,
    window: VecDeque<(DateTime<Utc>, f64)>,
    detector: AdaptiveTimeDetector,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BollingerBandsOutput {
    pub average: f64,
    pub upper: f64,
    pub lower: f64,
}

impl BollingerBands {
    pub fn get_window(&self) -> VecDeque<(DateTime<Utc>, f64)> {
        self.window.clone()
    }

    pub fn new(duration: Duration, multiplier: f64) -> Result<Self> {
        // Change: Check for zero duration (std::time::Duration can't be negative)
        if duration.as_secs() == 0 && duration.subsec_nanos() == 0 {
            return Err(crate::errors::TaError::InvalidParameter);
        }
        Ok(Self {
            duration,
            multiplier,
            sd: Sd::new(duration)?, // Pass std::time::Duration
            window: VecDeque::new(),
            detector: AdaptiveTimeDetector::new(duration),
        })
    }

    pub fn multiplier(&self) -> f64 {
        self.multiplier
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

impl Next<f64> for BollingerBands {
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

        // Calculate the mean and standard deviation based on the current window
        let values: Vec<f64> = self.window.iter().map(|&(_, val)| val).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let sd = self.sd.next((timestamp, value));
        mean + sd * self.multiplier
    }
}

impl Reset for BollingerBands {
    fn reset(&mut self) {
        self.sd.reset();
        self.window.clear();
        self.detector.reset();
    }
}

impl Default for BollingerBands {
    fn default() -> Self {
        // Change: Use Duration::from_secs for 14 days
        Self::new(Duration::from_secs(14 * 24 * 60 * 60), 2_f64).unwrap()
    }
}

impl fmt::Display for BollingerBands {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Change: Display duration in seconds
        write!(f, "BB({}s, {})", self.duration.as_secs(), self.multiplier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helper::*;
    use chrono::Utc;

    test_indicator!(BollingerBands);

    #[test]
    fn test_new() {
        // Change: Use std::time::Duration constructors
        assert!(BollingerBands::new(Duration::from_secs(0), 2_f64).is_err());
        assert!(BollingerBands::new(Duration::from_secs(86400), 2_f64).is_ok()); // 1 day
        assert!(BollingerBands::new(Duration::from_secs(172800), 2_f64).is_ok());
        // 2 days
    }

    #[test]
    fn test_next() {
        let mut bb = BollingerBands::new(Duration::from_secs(3 * 86400), 2.0).unwrap(); // 3 days
        let now = Utc::now();

        // Use chrono::Duration for date arithmetic
        let a = bb.next((now, 2.0));
        let b = bb.next((now + chrono::Duration::days(1), 5.0));
        let c = bb.next((now + chrono::Duration::days(2), 1.0));
        let d = bb.next((now + chrono::Duration::days(3), 6.25));

        assert_eq!(round(a), 2.0);
        assert_eq!(round(b), 6.5);
        assert_eq!(round(c), 6.066);
        assert_eq!(round(d), 8.562);
    }

    #[test]
    fn test_reset() {
        let mut bb = BollingerBands::new(Duration::from_secs(5 * 86400), 2.0_f64).unwrap(); // 5 days
        let now = Utc::now();

        let out = bb.next((now, 3.0));

        assert_eq!(out, 3.0);

        bb.next((now + chrono::Duration::days(1), 2.5));
        bb.next((now + chrono::Duration::days(2), 3.5));
        bb.next((now + chrono::Duration::days(3), 4.0));

        let out = bb.next((now + chrono::Duration::days(4), 2.0));

        assert_eq!(round(out), 4.414);

        bb.reset();
        let out = bb.next((now, 3.0));
        assert_eq!(out, 3.0);
    }

    #[test]
    fn test_default() {
        BollingerBands::default();
    }

    #[test]
    fn test_display() {
        let duration = Duration::from_secs(10 * 86400); // 10 days
        let bb = BollingerBands::new(duration, 3.0_f64).unwrap();
        assert_eq!(format!("{}", bb), format!("BB({}s, 3)", 10 * 86400));
    }
}
