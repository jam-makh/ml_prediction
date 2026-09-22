from dataclasses import dataclass
from typing import Iterator
import numpy as np
import pandas as pd


@dataclass
class ExpandingWindowSplit:
    """Expanding-window CV over a time column, not row positions.

    min_train : length of the first training window
    horizon   : length of each validation window
    step      : how far the validation window moves each fold
    gap       : embargo between train end and validation start (avoid leakage
                from lagged/rolling features that peek across the boundary)
    """
    min_train: pd.Timedelta
    horizon: pd.Timedelta
    step: pd.Timedelta | None = None
    gap: pd.Timedelta = pd.Timedelta(0)
    max_splits: int | None = None

    def split(self, df: pd.DataFrame, time_col: str) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        t = pd.to_datetime(df[time_col])
        start, end = t.min(), t.max()
        step = self.step or self.horizon

        train_end = start + self.min_train
        n = 0
        while True:
            val_start = train_end + self.gap
            val_end = val_start + self.horizon
            if val_end > end + pd.Timedelta(nanoseconds=1):
                break

            tr = np.flatnonzero((t >= start) & (t < train_end))
            va = np.flatnonzero((t >= val_start) & (t < val_end))
            if len(tr) and len(va):
                yield tr, va
                n += 1
                if self.max_splits and n >= self.max_splits:
                    break

            train_end = train_end + step
