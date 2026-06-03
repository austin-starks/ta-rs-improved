mod exponential_moving_average;
pub use self::exponential_moving_average::ExponentialMovingAverage;

mod simple_moving_average;
pub use self::simple_moving_average::SimpleMovingAverage;

mod standard_deviation;
pub use self::standard_deviation::StandardDeviation;

mod mean_absolute_deviation;
pub use self::mean_absolute_deviation::MeanAbsoluteDeviation;

mod relative_strength_index;
pub use self::relative_strength_index::RelativeStrengthIndex;

mod minimum;
pub use self::minimum::Minimum;

mod maximum;
pub use self::maximum::Maximum;

mod window_aggregate;

mod max_drawdown;
pub use self::max_drawdown::MaxDrawdown;

mod max_drawup;
pub use self::max_drawup::MaxDrawup;

mod bollinger_bands;
pub use self::bollinger_bands::{BollingerBands, BollingerBandsOutput};

mod rate_of_change;
pub use self::rate_of_change::RateOfChange;

mod adaptive;
pub use self::adaptive::{AdaptiveTimeDetector, DetectedFrequency};
