"""
TDSParams — central configuration for all TDS computations.

All parameters have defaults matching the original NCOM paper
(Bashan et al., Nature Communications 2012).
"""

from dataclasses import dataclass


@dataclass
class TDSParams:
    """
    Configuration for the TDS algorithm. Pass a single instance through
    all functions so settings are consistent end-to-end.

    Parameters
    ----------
    window : int
        Sliding window size in samples. Default 60 (= 60 s at 1 Hz).
    overlap : int
        Step size between windows in samples. Default 30 (50% overlap).
    max_lag : int
        Maximum cross-correlation lag to search, in samples (±max_lag).
        Default 30.
    stability_window : int
        Number of consecutive τ₀ points assessed for stability. Default 5.
    stability_min : int
        Minimum number of points within tolerance to label as stable.
        Default 4 (i.e. 4 out of 5).
    tolerance : int
        Allowed deviation in τ₀ (samples) to still be considered stable.
        Default ±1.
    n_surrogates : int
        Number of surrogate subjects for null distribution. Default 1000.
    alpha : float
        Significance level for threshold (1-alpha percentile). Default 0.05.
    """

    window: int = 60
    overlap: int = 30
    max_lag: int = 30
    stability_window: int = 5
    stability_min: int = 4
    tolerance: int = 1
    n_surrogates: int = 1000
    alpha: float = 0.05
