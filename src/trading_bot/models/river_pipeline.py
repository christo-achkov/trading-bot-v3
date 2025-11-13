"""Model factory utilities for online learning pipelines."""
from __future__ import annotations

from river import compose, drift, ensemble, metrics, preprocessing


def adaptive_classifier() -> compose.Pipeline:
    """Create the default adaptive classifier backed by an SRP ensemble."""

    estimator = ensemble.SRPClassifier(
        n_models=12,
        lam=8,
        drift_detector=drift.ADWIN(delta=0.002),
        seed=42,
    )
    return compose.Pipeline(
        preprocessing.StandardScaler(),
        estimator,
    )


def default_metrics() -> metrics.MultiClassClassificationMetric:
    """Provide metrics tracked during online evaluation."""

    return metrics.Accuracy()
