"""The train/holdout month cut, in plain Python.

Kept free of pandas so the pandas pipeline (``window.py``) and the Spark feature
build (``feature_eng_v3.py``) import the same rule.
"""

from __future__ import annotations

import math


def cut_positions(n_months: int, test_share: float, gap_months: int = 0) -> tuple[int, int]:
    """Return where training ends and the holdout starts in an ascending list of months.

    The holdout is the last ``floor(test_share * n_months)`` months, and the
    ``gap_months`` before it belong to neither side.

    Parameters
    ----------
    n_months : int
        Number of distinct months available.
    test_share : float
        Share of months held out, strictly between 0 and 1.
    gap_months : int, optional
        Months discarded between training and holdout. Default 0.

    Returns
    -------
    tuple of (int, int)
        ``(train_end, test_start)``: training is ``months[:train_end]`` and the
        holdout is ``months[test_start:]``.

    Raises
    ------
    ValueError
        If ``test_share`` is outside (0, 1), ``gap_months`` is negative, or
        either side would be empty.
    """
    if not 0 < test_share < 1:
        raise ValueError(f"split.test_share must be between 0 and 1, got {test_share}")
    if gap_months < 0:
        raise ValueError(f"split.gap_months cannot be negative, got {gap_months}")

    # Rounded before the floor, so float noise like 28.999999 still floors to 29.
    test_months = math.floor(round(n_months * test_share, 9))
    test_start = n_months - test_months
    train_end = test_start - gap_months
    if test_months < 1 or train_end < 1:
        raise ValueError(
            f"{n_months} months cannot hold a {test_share:.0%} holdout and a "
            f"{gap_months}-month gap with training months left over"
        )
    return train_end, test_start
