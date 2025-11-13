"""Model factory utilities for online learning pipelines."""
from __future__ import annotations

from river import compose, metrics, preprocessing
from river.drift import ADWIN
from river.drift.binary import DDM
from river.forest import ARFClassifier


def adaptive_classifier() -> compose.Pipeline:
    """Create an adaptive classifier with drift-aware ensemble retraining."""

    return compose.Pipeline(
        preprocessing.StandardScaler(),
        ARFClassifier(
            n_models=5,
            drift_detector=ADWIN(delta=1e-4),
            warning_detector=DDM(),
        ),
    )


def default_metrics() -> metrics.MultiClassClassificationMetric:
    """Provide metrics tracked during online evaluation."""

    return metrics.Accuracy()
