use std::collections::VecDeque;
use std::fmt;
use std::time::Duration;

use crate::errors::{Result, TaError};
use crate::indicators::AdaptiveTimeDetector;
use crate::{Next, NextBatch, Reset};
use chrono::{DateTime, Utc};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

const MAX_WINDOW_SIZE: usize = 500;
const KEEP_OLDEST: usize = 10;
const KEEP_RECENT: usize = 100;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Maximum {
    duration: Duration,
    window: VecDeque<(DateTime<Utc>, f64)>,
    detector: AdaptiveTimeDetector,
    /// Cached `chrono::Duration` form of `duration` (computed once on first use)
    /// so `next()` skips a `from_std` conversion every call. Not serialized;
    /// lazily recomputed after deserialization.
    #[cfg_attr(feature = "serde", serde(skip))]
    cached_window: Option<i64>,
    /// Monotonic-decreasing candidate deque mirroring `window`: entries run in
    /// increasing time (front oldest) and strictly decreasing value, so
    /// `front()` is always the max over the current window. Lets `next()` read
    /// the max in O(1) amortized instead of an O(window) `find_max_value` scan.
    /// Transient — rebuilt from `window` after a thin (which drops interior
    /// points) and defensively after a deserialize. Never a persisted contract
    /// (skipped in serde; the outer config re-derives ta state via warmup).
    #[cfg_attr(feature = "serde", serde(skip))]
    mono: VecDeque<(DateTime<Utc>, f64)>,
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
                cached_window: None,
                mono: VecDeque::new(),
            })
        }
    }

    /// Authoritative O(window) scan. Kept as the correctness oracle for the
    /// `debug_assert` in `next()`; compiled out of release builds.
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    fn find_max_value(&self) -> f64 {
        self.window
            .iter()
            .map(|&(_, val)| val)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Rebuild `mono` from the *sealed* points — every point except the current
    /// (newest) one, i.e. `window[..len-1]` — in O(n). Called after a thin
    /// (which drops arbitrary interior points) and defensively after a
    /// deserialize, where `mono` deserializes empty while `window` is populated.
    fn rebuild_mono(&mut self) {
        self.mono.clear();
        let sealed_len = self.window.len().saturating_sub(1);
        for &entry in self.window.iter().take(sealed_len) {
            while self.mono.back().map_or(false, |&(_, bv)| bv <= entry.1) {
                self.mono.pop_back();
            }
            self.mono.push_back(entry);
        }
    }

    fn remove_old_data(&mut self, current_time: DateTime<Utc>) {
        let dur_nanos = *self
            .cached_window
            .get_or_insert_with(|| self.duration.as_nanos() as i64);
        let cutoff_nanos = current_time.timestamp_nanos_opt().unwrap_or(i64::MIN) - dur_nanos;
        while self
            .window
            .front()
            .map_or(false, |(time, _)| time.timestamp_nanos_opt().unwrap_or(i64::MIN) <= cutoff_nanos)
        {
            self.window.pop_front();
        }
        // Evict the same expired points from the candidate deque (identical
        // predicate); `mono` is time-ordered front-oldest so this is O(evicted).
        while self
            .mono
            .front()
            .map_or(false, |(time, _)| time.timestamp_nanos_opt().unwrap_or(i64::MIN) <= cutoff_nanos)
        {
            self.mono.pop_front();
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

        for i in 0..middle_start.min(len) {
            new_window.push_back(self.window[i]);
        }

        let mut keep = true;
        for i in middle_start..middle_end {
            if keep {
                new_window.push_back(self.window[i]);
            }
            keep = !keep;
        }

        for i in middle_end..len {
            new_window.push_back(self.window[i]);
        }

        self.window = new_window;
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
        // Resync the transient candidate deque after a deserialize (window
        // populated from a snapshot, mono defaulted empty). Invariant otherwise:
        // mono is non-empty iff window has >= 2 points, so this fires only
        // post-deserialize.
        if self.mono.is_empty() && self.window.len() > 1 {
            self.rebuild_mono();
        }

        // Check if we should replace the last value (same time bucket)
        let should_replace = self.detector.should_replace(timestamp);

        // ALWAYS remove old data first, regardless of replace/add (evicts
        // expired points from both the window and the sealed-candidate deque).
        self.remove_old_data(timestamp);

        if should_replace {
            // Same bucket: drop the current (newest) point. It is `window.back()`
            // and is deliberately NOT in `mono` (which holds only the *sealed*
            // points — everything except the current one), so there is no
            // candidate-deque surgery to do. This is the O(1) intraday hot path.
            if !self.window.is_empty() {
                self.window.pop_back();
            }
        } else if let Some(&sealed) = self.window.back() {
            // New bucket: the point that was current becomes permanent. Seal it
            // into the monotonic deque now, dropping dominated tail candidates
            // (any tail value <= it can never again be the max while it is
            // in-window, since it is newer).
            while self.mono.back().map_or(false, |&(_, bv)| bv <= sealed.1) {
                self.mono.pop_back();
            }
            self.mono.push_back(sealed);
        }

        // The new point becomes the current (newest) point. It stays OUT of
        // `mono` until a later new-bucket tick seals it.
        self.window.push_back((timestamp, value));

        // Thin window if it exceeds max size (sparse sampling for memory
        // efficiency). Thinning drops interior points, so rebuild mono to match.
        let len_before = self.window.len();
        self.thin_window();
        if self.window.len() != len_before {
            self.rebuild_mono();
        }

        // O(1) max = max(best sealed candidate, current point). In debug builds,
        // cross-check against the authoritative scan so any desync fails loudly
        // under the existing + differential tests.
        let max = match (self.mono.front(), self.window.back()) {
            (Some(&(_, s)), Some(&(_, c))) => s.max(c),
            (None, Some(&(_, c))) => c,
            (Some(&(_, s)), None) => s,
            (None, None) => f64::NEG_INFINITY,
        };
        debug_assert_eq!(
            max,
            self.find_max_value(),
            "monotonic max desynced from window scan"
        );
        max
    }
}

impl NextBatch<f64> for Maximum {}

impl Reset for Maximum {
    fn reset(&mut self) {
        self.window.clear();
        self.mono.clear();
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

    // Deterministic LCG so the property tests are reproducible without a dev-dep.
    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state >> 33
    }

    /// The O(1) monotonic path must equal the O(window) scan for every call.
    /// `next()` already `debug_assert`s `mono.front() == find_max_value()` on
    /// each step, so feeding varied/adversarial sequences here makes that scan
    /// oracle validate the fast path across eviction, same-bucket replacement,
    /// thinning (>500 points), equal values, and monotonic up/down runs.
    #[test]
    fn monotonic_matches_scan_over_random_sequences() {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        // Mix sub-daily and multi-day windows; the detector's bucket mode and
        // thus same-bucket replacement differ across these durations.
        for &secs in &[2u64, 3600, 86_400, 7 * 86_400] {
            let mut max = Maximum::new(Duration::from_secs(secs)).unwrap();
            let mut t = Utc.ymd(2020, 1, 1).and_hms(0, 0, 0);
            for _ in 0..1500 {
                // Step 0..=2*window: 0 forces same-timestamp replaces, large
                // steps force eviction; everything between exercises the mix.
                let step = (lcg(&mut state) % (secs * 2 + 1)) as i64;
                t = t + chrono::Duration::seconds(step);
                let v = (lcg(&mut state) % 20_000) as f64 / 100.0 - 100.0; // [-100,100)
                let _ = max.next((t, v)); // internal debug_assert is the oracle
            }
        }
    }

    /// Force >500 in-window points (huge window, no eviction) so `thin_window`
    /// fires repeatedly and `mono` is rebuilt from the thinned window. The
    /// internal debug_assert validates `mono.front() == find_max_value()` over
    /// the (thinned) window on every step — i.e. the O(1) path stays identical
    /// to the scan the original used, INCLUDING thinning's approximation, which
    /// we deliberately preserve. So we bound rather than assert exactness: the
    /// thinned max can never exceed the true running max, and stays finite.
    #[test]
    fn monotonic_matches_scan_across_thinning() {
        let mut max = Maximum::new(Duration::from_secs(1_000_000 * 86_400)).unwrap();
        let start = Utc.ymd(2000, 1, 1).and_hms(0, 0, 0);
        let mut true_running_max = f64::NEG_INFINITY;
        for i in 0..900i64 {
            // Scattered values so the max can land on interior points thinning drops.
            let v = ((i.wrapping_mul(2_654_435_761)) % 1000) as f64;
            true_running_max = true_running_max.max(v);
            let got = max.next((start + chrono::Duration::seconds(i), v));
            assert!(got.is_finite());
            assert!(got <= true_running_max, "thinned max exceeded true max");
        }
    }
}
