use std::collections::VecDeque;
use std::fmt;
use std::time::Duration; // Change: Use std::time::Duration

use crate::errors::Result;
use crate::indicators::{AdaptiveTimeDetector, ExponentialMovingAverage as Ema};
use crate::{Next, Reset};
use chrono::{DateTime, Utc};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[doc(alias = "RSI")]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct RelativeStrengthIndex {
    duration: Duration, // Now std::time::Duration
    up_ema_indicator: Ema,
    down_ema_indicator: Ema,
    window: VecDeque<(DateTime<Utc>, f64)>,
    prev_val: Option<f64>,
    detector: AdaptiveTimeDetector,
}

impl RelativeStrengthIndex {
    pub fn new(duration: Duration) -> Result<Self> {
        // Note: Ema::new() now also expects std::time::Duration
        Ok(Self {
            duration,
            up_ema_indicator: Ema::new(duration)?, // Assuming 14-period EMA
            down_ema_indicator: Ema::new(duration)?,
            window: VecDeque::new(),
            prev_val: None,
            detector: AdaptiveTimeDetector::new(duration),
        })
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

impl Next<f64> for RelativeStrengthIndex {
    type Output = f64;

    fn next(&mut self, (timestamp, value): (DateTime<Utc>, f64)) -> Self::Output {
        // Check if we should replace the last value (same time bucket)
        let should_replace = self.detector.should_replace(timestamp);

        // ALWAYS remove old data first, regardless of replace/add
        self.remove_old_data(timestamp);

        if should_replace && !self.window.is_empty() {
            // For RSI, when replacing a value in the same time bucket,
            // we don't change prev_val since it represents the previous period's close
            // Just remove the last window entry to be replaced
            self.window.pop_back();
        } else {
            // Update prev_val to the last complete period's value
            // This is crucial: prev_val should be the closing value of the previous period
            if !self.window.is_empty() {
                self.prev_val = Some(self.window.back().unwrap().1);
            }
        }

        // Calculate gain and loss using the stable prev_val
        let (gain, loss) = if let Some(prev_val) = self.prev_val {
            if value > prev_val {
                (value - prev_val, 0.0)
            } else {
                (0.0, prev_val - value)
            }
        } else {
            (0.0, 0.0)
        };

        // Add to window AFTER calculating gain/loss
        self.window.push_back((timestamp, value));

        // Only update prev_val for the NEXT period if this is not a replacement
        // When replacing, prev_val stays as the previous period's close
        if !should_replace {
            self.prev_val = Some(value);
        }

        // Update EMAs
        let avg_up = self.up_ema_indicator.next((timestamp, gain));
        let avg_down = self.down_ema_indicator.next((timestamp, loss));

        // Calculate and return RSI
        if avg_down == 0.0 {
            if avg_up == 0.0 {
                50.0 // Neutral value when no movement
            } else {
                100.0 // Max value when only gains
            }
        } else {
            let rs = avg_up / avg_down;
            100.0 - (100.0 / (1.0 + rs))
        }
    }
}

impl Reset for RelativeStrengthIndex {
    fn reset(&mut self) {
        self.window.clear();
        self.prev_val = None;
        self.up_ema_indicator.reset();
        self.down_ema_indicator.reset();
        self.detector.reset();
    }
}

impl Default for RelativeStrengthIndex {
    fn default() -> Self {
        // Change: Use Duration::from_secs for 14 days
        Self::new(Duration::from_secs(14 * 24 * 60 * 60)).unwrap()
    }
}

impl fmt::Display for RelativeStrengthIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Change: Calculate days from seconds
        let days = self.duration.as_secs() / 86400;
        write!(f, "RSI({} days)", days)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helper::*;
    use chrono::{TimeZone, Utc};

    test_indicator!(RelativeStrengthIndex);

    #[test]
    fn test_new() {
        // Change: Use std::time::Duration constructors
        assert!(RelativeStrengthIndex::new(Duration::from_secs(0)).is_err());
        assert!(RelativeStrengthIndex::new(Duration::from_secs(86400)).is_ok());
        // 1 day
    }

    #[test]
    fn test_next() {
        let mut rsi = RelativeStrengthIndex::new(Duration::from_secs(3 * 86400)).unwrap(); // 3 days
        let timestamp = Utc.ymd(2020, 1, 1).and_hms(0, 0, 0);

        // First value: 10.0 (no previous value, so RSI = 50)
        assert_eq!(rsi.next((timestamp, 10.0)), 50.0);

        // Second value: 10.5 (gain of 0.5, no loss)
        assert_eq!(
            rsi.next((timestamp + chrono::Duration::days(1), 10.5))
                .round(),
            100.0
        );

        // Third value: 10.0 (loss of 0.5 from 10.5)
        // With EMA k=0.5: avg_up=0.125, avg_down=0.25, RS=0.5, RSI=33.33
        assert_eq!(
            rsi.next((timestamp + chrono::Duration::days(2), 10.0))
                .round(),
            33.0
        );

        // Fourth value: 9.5 (loss of 0.5 from 10.0)
        // With continued losses, RSI should drop further
        // avg_up = 0.0625, avg_down = 0.375, RS = 0.1667, RSI = 14.3
        assert_eq!(
            rsi.next((timestamp + chrono::Duration::days(3), 9.5))
                .round(),
            14.0
        );
    }

    #[test]
    fn test_reset() {
        let mut rsi = RelativeStrengthIndex::new(Duration::from_secs(3 * 86400)).unwrap(); // 3 days
        let timestamp = Utc.ymd(2020, 1, 1).and_hms(0, 0, 0);
        assert_eq!(rsi.next((timestamp, 10.0)), 50.0);
        assert_eq!(
            rsi.next((timestamp + chrono::Duration::days(1), 10.5))
                .round(),
            100.0
        );

        rsi.reset();
        assert_eq!(rsi.next((timestamp, 10.0)).round(), 50.0);
        assert_eq!(
            rsi.next((timestamp + chrono::Duration::days(1), 10.5))
                .round(),
            100.0
        );
    }

    #[test]
    fn test_default() {
        RelativeStrengthIndex::default();
    }

    #[test]
    fn test_display() {
        let rsi = RelativeStrengthIndex::new(Duration::from_secs(16 * 86400)).unwrap(); // 16 days
        assert_eq!(format!("{}", rsi), "RSI(16 days)");
    }
}
