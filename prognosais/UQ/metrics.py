"""Streaming entropy metrics used by uncertainty-quantification methods."""

from __future__ import annotations

import numpy as np


class PredictiveEntropy:
    """Accumulate model probabilities and compute entropy of their mean."""

    def __init__(self, class_axis: int, epsilon: float) -> None:
        """Initialize an empty predictive-entropy accumulator.

        Args:
            class_axis: Axis containing class probabilities.
            epsilon: Positive numerical-stability constant for logarithms.

        Raises:
            ValueError: If epsilon is not strictly positive.
        """

        if epsilon <= 0:
            raise ValueError("epsilon must be strictly positive.")
        self.sum_probs: np.ndarray | None = None
        self.total_models = 0
        self.class_axis = int(class_axis)
        self.epsilon = float(epsilon)

    def update(self, model_probs: np.ndarray) -> None:
        """Add probabilities from one stochastic pass or ensemble member.

        The first input is copied so subsequent updates never mutate an array
        owned by the caller.

        Args:
            model_probs: Probability tensor with classes on class_axis.

        Raises:
            ValueError: If an update has a different shape or contains
                non-finite values.
        """

        probabilities = np.asarray(model_probs)
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("Predictive-entropy probabilities must be finite.")
        if self.sum_probs is None:
            self.sum_probs = np.array(probabilities, copy=True)
        else:
            if probabilities.shape != self.sum_probs.shape:
                raise ValueError(
                    "Predictive-entropy update shape mismatch: "
                    f"expected {self.sum_probs.shape}, got {probabilities.shape}."
                )
            self.sum_probs += probabilities
        self.total_models += 1

    def compute(self) -> np.ndarray:
        """Compute entropy of the mean probability tensor.

        Returns:
            Predictive entropy with the configured class axis removed.

        Raises:
            ValueError: If no probabilities have been accumulated.
        """

        if self.total_models == 0 or self.sum_probs is None:
            raise ValueError(
                "No models processed. Update with model probabilities first."
            )

        mean_probs = self.sum_probs / float(self.total_models)
        log_probs = -np.log(mean_probs + self.epsilon)
        return np.sum(mean_probs * log_probs, axis=self.class_axis)


class ExpectedEntropy:
    """Accumulate per-member entropy and compute its ensemble expectation."""

    def __init__(self, class_axis: int, epsilon: float) -> None:
        """Initialize an empty expected-entropy accumulator.

        Args:
            class_axis: Axis containing class probabilities.
            epsilon: Positive numerical-stability constant for logarithms.

        Raises:
            ValueError: If epsilon is not strictly positive.
        """

        if epsilon <= 0:
            raise ValueError("epsilon must be strictly positive.")
        self.running_sum: np.ndarray | None = None
        self.total_models = 0
        self.class_axis = int(class_axis)
        self.epsilon = float(epsilon)

    def update(self, model_probs: np.ndarray) -> None:
        """Add the entropy produced by one probability tensor.

        Args:
            model_probs: Probability tensor with classes on class_axis.

        Raises:
            ValueError: If an update has a different entropy-map shape or
                contains non-finite values.
        """

        probabilities = np.asarray(model_probs)
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("Expected-entropy probabilities must be finite.")
        weighted_log_probs = np.sum(
            probabilities * -np.log(probabilities + self.epsilon),
            axis=self.class_axis,
        )

        if self.running_sum is None:
            self.running_sum = np.array(weighted_log_probs, copy=True)
        else:
            if weighted_log_probs.shape != self.running_sum.shape:
                raise ValueError(
                    "Expected-entropy update shape mismatch: "
                    f"expected {self.running_sum.shape}, "
                    f"got {weighted_log_probs.shape}."
                )
            self.running_sum += weighted_log_probs
        self.total_models += 1

    def compute(self) -> np.ndarray:
        """Compute mean entropy across accumulated members.

        Returns:
            Expected entropy with the configured class axis removed.

        Raises:
            ValueError: If no probabilities have been accumulated.
        """

        if self.total_models == 0 or self.running_sum is None:
            raise ValueError(
                "No models processed. Update with model probabilities first."
            )
        return self.running_sum / float(self.total_models)
