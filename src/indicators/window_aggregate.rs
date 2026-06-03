//! Sliding-window aggregation (SWAG) over an associative monoid.
//!
//! Used by `MaxDrawdown` / `MaxDrawup` to replace their O(W)-per-bar full
//! window rescan with amortized O(1)-per-op maintenance, while producing
//! bit-identical output (see the randomized equivalence tests in each
//! indicator). The aggregates assume strictly positive inputs (prices /
//! equity), which is the only domain the indicators are fed.

/// A segment aggregate that forms a monoid under [`combine`](WindowAggregate::combine).
/// `self` is the older (left) segment, `other` the newer (right) one; order matters.
pub trait WindowAggregate: Clone {
    fn leaf(value: f64) -> Self;
    fn combine(&self, other: &Self) -> Self;
    /// The metric the indicator reports, as a 0..1 ratio (pre-`*100`).
    fn ratio(&self) -> f64;
}

/// Two-stack sliding-window aggregation supporting `push_back` + `pop_front`
/// in amortized O(1). Each stack entry caches the running aggregate so the
/// whole-window aggregate is one `combine` away.
#[derive(Debug, Clone)]
pub struct MonoidWindow<A: WindowAggregate> {
    front: Vec<(f64, A)>, // pop side; top = oldest, agg covers self..bottom (old->new)
    back: Vec<(f64, A)>,  // push side; top = newest, agg covers bottom..self (old->new)
}

impl<A: WindowAggregate> Default for MonoidWindow<A> {
    fn default() -> Self {
        Self {
            front: Vec::new(),
            back: Vec::new(),
        }
    }
}

impl<A: WindowAggregate> MonoidWindow<A> {
    pub fn clear(&mut self) {
        self.front.clear();
        self.back.clear();
    }

    pub fn push_back(&mut self, value: f64) {
        let leaf = A::leaf(value);
        let agg = match self.back.last() {
            Some((_, below)) => below.combine(&leaf),
            None => leaf,
        };
        self.back.push((value, agg));
    }

    pub fn pop_front(&mut self) -> Option<f64> {
        if self.front.is_empty() {
            // Rotate `back` (old->new bottom->top) into `front` so its top is oldest.
            let mut acc: Option<A> = None;
            while let Some((value, _)) = self.back.pop() {
                let leaf = A::leaf(value);
                let agg = match &acc {
                    Some(below) => leaf.combine(below),
                    None => leaf,
                };
                acc = Some(agg.clone());
                self.front.push((value, agg));
            }
        }
        self.front.pop().map(|(value, _)| value)
    }

    /// Whole-window aggregate (`front` is older, `back` is newer), or `None` if empty.
    pub fn aggregate(&self) -> Option<A> {
        match (self.front.last(), self.back.last()) {
            (Some((_, f)), Some((_, b))) => Some(f.combine(b)),
            (Some((_, f)), None) => Some(f.clone()),
            (None, Some((_, b))) => Some(b.clone()),
            (None, None) => None,
        }
    }
}

/// Maximum drawdown aggregate: peak-to-trough decline, peak before trough.
#[derive(Debug, Clone)]
pub struct DrawdownAgg {
    min: f64,
    max: f64,
    ratio: f64,
}

impl WindowAggregate for DrawdownAgg {
    fn leaf(value: f64) -> Self {
        Self {
            min: value,
            max: value,
            ratio: 0.0,
        }
    }

    fn combine(&self, other: &Self) -> Self {
        let cross = (self.max - other.min) / self.max;
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
            ratio: self.ratio.max(other.ratio).max(cross),
        }
    }

    fn ratio(&self) -> f64 {
        self.ratio
    }
}

/// Maximum drawup aggregate: trough-to-peak rise, trough before peak.
#[derive(Debug, Clone)]
pub struct DrawupAgg {
    min: f64,
    max: f64,
    ratio: f64,
}

impl WindowAggregate for DrawupAgg {
    fn leaf(value: f64) -> Self {
        Self {
            min: value,
            max: value,
            ratio: 0.0,
        }
    }

    fn combine(&self, other: &Self) -> Self {
        let cross = (other.max - self.min) / self.min;
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
            ratio: self.ratio.max(other.ratio).max(cross),
        }
    }

    fn ratio(&self) -> f64 {
        self.ratio
    }
}
