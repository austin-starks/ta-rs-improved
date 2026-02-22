use chrono::{DateTime, Utc};
use std::time::Duration;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Represents the frequency mode for de-duplication
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum DetectedFrequency {
    /// Still learning from initial data points (kept for backward compatibility only)
    Unknown,
    /// Daily mode: maintains 3.4 hour gap between points
    DailyOHLC,
    /// Intraday mode: minute-level bucketing
    Intraday(Duration),
}

/// Handles time-based de-duplication logic for indicators
///
/// Uses a simple duration-based rule:
/// - Indicators with duration < 1 day: Use minute-level bucketing (replace within same minute)
/// - Indicators with duration >= 1 day: Use daily bucketing with 3.4 hour gap enforcement
///
/// The 3.4 hour gap for daily indicators ensures:
/// - Half-day sessions (~3.5 hours): Captures both open and close as separate points
/// - Full-day sessions (~6.5 hours): Captures morning and afternoon as separate points
/// - Minutely updates during market hours: Continuously updates the current slot
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct AdaptiveTimeDetector {
    frequency: DetectedFrequency,
    last_minute_bucket: i64,
    last_timestamp: Option<DateTime<Utc>>,
}

impl AdaptiveTimeDetector {
    /// Create a new detector for a specific indicator duration
    ///
    /// # Arguments
    /// * `duration` - The indicator's time window duration, used to determine bucketing strategy:
    ///   - Duration < 5 minutes: Uses second-level bucketing
    ///   - Duration < 1 day: Uses minute-level bucketing
    ///   - Duration >= 1 day: Uses daily bucketing with 3.4 hour gap enforcement
    pub fn new(duration: Duration) -> Self {
        let frequency = if duration < Duration::from_secs(5 * 60) {
            // Use second bucketing for very short-term indicators (< 5 minutes)
            DetectedFrequency::Intraday(Duration::from_secs(1))
        } else if duration < Duration::from_secs(86400) {
            // Use minute bucketing for short-term indicators (< 1 day)
            DetectedFrequency::Intraday(Duration::from_secs(60))
        } else {
            // Use daily bucketing with gap enforcement for long-term indicators
            DetectedFrequency::DailyOHLC
        };

        Self {
            frequency,
            last_minute_bucket: i64::MIN,
            last_timestamp: None,
        }
    }

    /// Create a new detector with custom detection samples (DEPRECATED - use new())
    #[deprecated(since = "1.0.0", note = "Use new(duration) instead")]
    pub fn with_samples(_detection_samples: usize, duration: Duration) -> Self {
        Self::new(duration)
    }

    /// Get the current frequency mode
    pub fn frequency(&self) -> &DetectedFrequency {
        &self.frequency
    }

    /// Process a new timestamp and determine if it should replace the previous value
    /// Returns true if this is a duplicate within the same time bucket (should replace)
    /// Returns false if this is a new time period (should append)
    pub fn should_replace(&mut self, timestamp: DateTime<Utc>) -> bool {
        match &self.frequency {
            DetectedFrequency::Intraday(bucket_duration) => {
                // Dynamic bucketing based on bucket_duration (second or minute level)
                let bucket_seconds = bucket_duration.as_secs() as i64;
                let current_bucket = timestamp.timestamp() / bucket_seconds;

                // Check if we're in the same bucket as last processed
                let should_replace = current_bucket == self.last_minute_bucket;

                // Update last processed bucket
                self.last_minute_bucket = current_bucket;

                should_replace
            }
            DetectedFrequency::DailyOHLC => {
                // Daily bucketing with 3.4 hour gap enforcement
                // The 3.4 hour threshold ensures:
                // - Half-day sessions (~3.5 hours): Captures both open and close
                // - Full-day sessions (~6.5 hours): Captures morning and afternoon

                // 3.4 hours = 3 hours 24 minutes = 12,240 seconds
                let min_gap = chrono::Duration::seconds(3 * 3600 + 24 * 60);

                if let Some(last_ts) = self.last_timestamp {
                    let time_diff = timestamp - last_ts;

                    // If within 3.4 hours of the last point, replace it
                    // This handles minutely data during market hours
                    if time_diff > chrono::Duration::zero() && time_diff < min_gap {
                        // Don't update last_timestamp here - we're replacing
                        return true;
                    }
                }

                // Either first point or more than 3.4 hours have passed
                // This is a new slot, so update last_timestamp
                self.last_timestamp = Some(timestamp);
                false
            }
            DetectedFrequency::Unknown => {
                // Shouldn't happen but default to not replacing
                self.last_timestamp = Some(timestamp);
                false
            }
        }
    }

    /// Reset the detector to initial state
    pub fn reset(&mut self) {
        self.last_minute_bucket = i64::MIN;
        self.last_timestamp = None;
        // Keep frequency as it was set based on duration
    }

    /// Check if frequency has been detected
    /// Always returns true since we determine mode immediately from duration
    pub fn is_detected(&self) -> bool {
        self.frequency != DetectedFrequency::Unknown
    }
}

// Default implementation removed - force explicit duration
// impl Default for AdaptiveTimeDetector {
//     fn default() -> Self {
//         Self::new()
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_daily_indicator_with_gap_enforcement() {
        let mut detector = AdaptiveTimeDetector::new(Duration::from_secs(2 * 86400)); // 2 days
        assert_eq!(detector.frequency(), &DetectedFrequency::DailyOHLC);

        let base = Utc.ymd(2024, 1, 1).and_hms(9, 30, 0);

        // First point
        assert!(!detector.should_replace(base));

        // Point 6.5 hours later (full trading day) - new slot
        assert!(!detector
            .should_replace(base + chrono::Duration::hours(6) + chrono::Duration::minutes(30)));

        // Next day - new slot
        assert!(!detector.should_replace(base + chrono::Duration::days(1)));

        // Now test the 3.4 hour gap enforcement
        let market_open = base + chrono::Duration::days(2);
        assert!(!detector.should_replace(market_open)); // New day, new slot

        // Minutely updates within 3.4 hours should replace
        assert!(detector.should_replace(market_open + chrono::Duration::minutes(1)));
        assert!(detector.should_replace(market_open + chrono::Duration::minutes(30)));
        assert!(detector.should_replace(market_open + chrono::Duration::hours(3)));

        // After 3.4 hours, should create new slot
        assert!(!detector.should_replace(
            market_open + chrono::Duration::hours(3) + chrono::Duration::minutes(25)
        ));
    }

    #[test]
    fn test_half_day_trading() {
        let mut detector = AdaptiveTimeDetector::new(Duration::from_secs(3 * 86400)); // 3 days
        assert_eq!(detector.frequency(), &DetectedFrequency::DailyOHLC);

        // Simulate half-day trading (market closes at 1:00 PM, ~3.5 hours)
        let half_day_open = Utc.ymd(2024, 1, 2).and_hms(9, 30, 0);
        assert!(!detector.should_replace(half_day_open)); // New slot for open price

        // All updates during first 3.4 hours should replace the open price
        assert!(detector.should_replace(half_day_open + chrono::Duration::minutes(30)));
        assert!(detector.should_replace(half_day_open + chrono::Duration::hours(2)));
        assert!(detector.should_replace(half_day_open + chrono::Duration::hours(3)));
        assert!(detector.should_replace(
            half_day_open + chrono::Duration::hours(3) + chrono::Duration::minutes(20)
        )); // Still within 3.4 hours

        // Close at 1:00 PM (3.5 hours) should create NEW slot since 3.5 > 3.4
        // This captures the closing price as a separate data point
        let half_day_close =
            half_day_open + chrono::Duration::hours(3) + chrono::Duration::minutes(30);
        assert!(!detector.should_replace(half_day_close)); // New slot for close price

        // If more updates come in near close, they replace the close price
        assert!(detector.should_replace(half_day_close + chrono::Duration::minutes(1)));

        // Next day should be a new slot
        let next_day = half_day_open + chrono::Duration::days(1);
        assert!(!detector.should_replace(next_day));
    }

    #[test]
    fn test_intraday_indicator() {
        let mut detector = AdaptiveTimeDetector::new(Duration::from_secs(15 * 60)); // 15 minutes
        assert!(matches!(
            detector.frequency(),
            DetectedFrequency::Intraday(d) if d.as_secs() == 60
        ));

        let base = Utc.ymd(2024, 1, 1).and_hms(9, 30, 0);

        // First data point
        assert!(!detector.should_replace(base));

        // Same minute - should replace
        assert!(detector.should_replace(base + chrono::Duration::seconds(30)));

        // Next minute - new slot
        assert!(!detector.should_replace(base + chrono::Duration::minutes(1)));

        // Within same minute - should replace
        assert!(detector
            .should_replace(base + chrono::Duration::minutes(1) + chrono::Duration::seconds(15)));

        // Next minute - new slot
        assert!(!detector.should_replace(base + chrono::Duration::minutes(2)));
    }

    #[test]
    fn test_full_trading_day_with_minutely_updates() {
        let mut detector = AdaptiveTimeDetector::new(Duration::from_secs(5 * 86400)); // 5 days

        // Full trading day: 9:30 AM to 4:00 PM (6.5 hours)
        let market_open = Utc.ymd(2024, 1, 2).and_hms(9, 30, 0);

        // First data point at market open
        assert!(!detector.should_replace(market_open));

        // Minutely updates for first 3 hours should all replace
        for minutes in 1..=180 {
            assert!(
                detector.should_replace(market_open + chrono::Duration::minutes(minutes)),
                "Should replace at {} minutes after open",
                minutes
            );
        }

        // After 3.4 hours (204 minutes), should create new slot
        assert!(!detector.should_replace(market_open + chrono::Duration::minutes(205)));

        // Subsequent updates should replace this new slot
        assert!(detector.should_replace(market_open + chrono::Duration::minutes(210)));
        assert!(detector.should_replace(
            market_open + chrono::Duration::hours(6) // Near market close
        ));
    }

    #[test]
    fn test_reset() {
        let mut detector = AdaptiveTimeDetector::new(Duration::from_secs(86400));
        let base = Utc.ymd(2024, 1, 1).and_hms(9, 30, 0);

        // Use detector
        detector.should_replace(base);
        detector.should_replace(base + chrono::Duration::hours(1));

        // Reset
        detector.reset();

        // Frequency should remain but state should be cleared
        assert_eq!(detector.frequency(), &DetectedFrequency::DailyOHLC);
        assert!(detector.last_timestamp.is_none());

        // Should work normally after reset
        assert!(!detector.should_replace(base + chrono::Duration::days(1)));
    }

    #[test]
    fn test_memory_footprint_for_trading_sessions() {
        let mut detector = AdaptiveTimeDetector::new(Duration::from_secs(10 * 86400)); // 10 days

        // Track which timestamps would be kept (not replaced)
        let mut kept_timestamps = Vec::new();

        // Half-day: Should keep exactly 2 points (open and close)
        let half_day_open = Utc.ymd(2024, 1, 2).and_hms(9, 30, 0);
        if !detector.should_replace(half_day_open) {
            kept_timestamps.push(half_day_open);
        }

        // Minutely updates that get replaced
        for minutes in 1..209 {
            let ts = half_day_open + chrono::Duration::minutes(minutes);
            if !detector.should_replace(ts) {
                kept_timestamps.push(ts);
            }
        }

        // Close at 1:00 PM (210 minutes = 3.5 hours)
        let half_day_close = half_day_open + chrono::Duration::minutes(210);
        if !detector.should_replace(half_day_close) {
            kept_timestamps.push(half_day_close);
        }

        // We should have exactly 2 timestamps for the half-day
        assert_eq!(
            kept_timestamps.len(),
            2,
            "Half-day should keep exactly 2 points"
        );

        // Full trading day test
        kept_timestamps.clear();
        detector.reset();

        let full_day_open = Utc.ymd(2024, 1, 3).and_hms(9, 30, 0);
        if !detector.should_replace(full_day_open) {
            kept_timestamps.push(full_day_open);
        }

        // Minutely updates for 6.5 hours (390 minutes)
        for minutes in 1..=390 {
            let ts = full_day_open + chrono::Duration::minutes(minutes);
            if !detector.should_replace(ts) {
                kept_timestamps.push(ts);
            }
        }

        // We should have exactly 2 timestamps for the full day
        // One at open, one after 3.4 hours (204 minutes)
        assert_eq!(
            kept_timestamps.len(),
            2,
            "Full day should keep exactly 2 points"
        );

        // Verify the gap between kept points is > 3.4 hours
        let gap = kept_timestamps[1] - kept_timestamps[0];
        assert!(
            gap >= chrono::Duration::minutes(204),
            "Gap between points should be >= 3.4 hours"
        );
    }
}
