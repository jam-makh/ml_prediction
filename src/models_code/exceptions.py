"""Custom exceptions raised by the models."""

from __future__ import annotations


class NotFittedError(RuntimeError):
    """Raised when a model is asked to predict or save before it has been fitted.

    A distinct type, so a caller can tell it apart from a failure inside a fitted model.
    """
