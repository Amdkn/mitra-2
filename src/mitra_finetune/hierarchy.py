"""Balanced many-class hierarchy for fixed-width classification heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class NodePrediction:
    """Validation/test probabilities returned by one native-head node."""

    validation_probabilities: np.ndarray
    test_probabilities: np.ndarray


@dataclass(frozen=True)
class HierarchicalPrediction:
    """Full-class validation/test probabilities."""

    validation_probabilities: np.ndarray
    test_probabilities: np.ndarray


NodeRunner = Callable[..., NodePrediction]


def _take_rows(X, mask: np.ndarray):
    indices = np.flatnonzero(mask)
    if hasattr(X, "iloc"):
        return X.iloc[indices]
    return np.asarray(X)[indices]


def _normalized_probabilities(
    probabilities: np.ndarray,
    *,
    n_rows: int,
    n_classes: int,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    expected = (n_rows, n_classes)
    if probabilities.shape != expected:
        raise ValueError(
            f"Expected node probabilities shaped {expected}, got "
            f"{probabilities.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("Hierarchy node returned non-finite probabilities")
    if (probabilities < -1e-7).any():
        raise ValueError("Hierarchy node returned negative probabilities")

    probabilities = np.maximum(probabilities, 0.0)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if (row_sums <= 0.0).any():
        raise ValueError("Hierarchy node returned a zero-sum probability row")
    return probabilities / row_sums


def hierarchical_predict_proba(
    *,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    node_runner: NodeRunner,
    max_classes: int,
    random_state: int,
) -> HierarchicalPrediction:
    """Fit a balanced class tree and compose full-class probabilities.

    ``node_runner`` receives one node's train/validation data plus the full
    validation and test query tables. It must return probabilities over that
    node's outputs for both full query tables.
    """

    if max_classes < 2:
        raise ValueError("max_classes must be at least 2")

    y_train = np.asarray(y_train).reshape(-1)
    y_val = np.asarray(y_val).reshape(-1)
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train have different lengths")
    if len(X_val) != len(y_val):
        raise ValueError("X_val and y_val have different lengths")
    if len(getattr(X_train, "shape", ())) != 2:
        raise ValueError("X_train must be two-dimensional")
    if len(getattr(X_val, "shape", ())) != 2:
        raise ValueError("X_val must be two-dimensional")
    if len(getattr(X_test, "shape", ())) != 2:
        raise ValueError("X_test must be two-dimensional")
    if not (
        X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    ):
        raise ValueError("Train, validation, and test feature counts differ")

    classes = np.unique(y_train)
    if len(classes) <= max_classes:
        raise ValueError(
            "Hierarchy is only required above the native class limit"
        )
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError(
            "Many-class labels must be contiguous integers starting at zero"
        )
    if not np.array_equal(np.unique(y_val), classes):
        raise ValueError(
            "Many-class validation must contain every training class"
        )
    y_train_encoded = y_train.astype(np.int64, copy=False)
    y_val_encoded = y_val.astype(np.int64, copy=False)

    ordered_classes = np.random.default_rng(random_state).permutation(
        np.arange(len(classes))
    )
    node_count = 0

    def fit_predict_node(
        node_classes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nonlocal node_count

        node_classes = np.asarray(node_classes, dtype=np.int64)
        if len(node_classes) == 1:
            return (
                np.ones((len(X_val), 1), dtype=np.float64),
                np.ones((len(X_test), 1), dtype=np.float64),
                node_classes,
            )

        train_mask = np.isin(y_train_encoded, node_classes)
        val_mask = np.isin(y_val_encoded, node_classes)
        node_y_train_global = y_train_encoded[train_mask]
        node_y_val_global = y_val_encoded[val_mask]

        is_leaf = len(node_classes) <= max_classes
        if is_leaf:
            outputs = tuple(np.asarray([value]) for value in node_classes)
        else:
            n_groups = min(
                int(np.ceil(len(node_classes) / max_classes)),
                max_classes,
            )
            outputs = tuple(
                group
                for group in np.array_split(node_classes, n_groups)
                if len(group)
            )
        if not 2 <= len(outputs) <= max_classes:
            raise RuntimeError(
                f"Hierarchy node produced {len(outputs)} outputs; expected "
                f"between 2 and {max_classes}"
            )

        output_by_class = {
            int(class_id): output_index
            for output_index, output_classes in enumerate(outputs)
            for class_id in output_classes
        }
        node_y_train = np.asarray(
            [output_by_class[int(value)] for value in node_y_train_global],
            dtype=np.int64,
        )
        node_y_val = np.asarray(
            [output_by_class[int(value)] for value in node_y_val_global],
            dtype=np.int64,
        )
        expected_outputs = np.arange(len(outputs))
        if not np.array_equal(np.unique(node_y_train), expected_outputs):
            raise ValueError("A hierarchy training node lost an output class")
        if not np.array_equal(np.unique(node_y_val), expected_outputs):
            raise ValueError("A hierarchy validation node lost an output class")

        node_index = node_count
        prediction = node_runner(
            X_train=_take_rows(X_train, train_mask),
            y_train=node_y_train,
            X_val=_take_rows(X_val, val_mask),
            y_val=node_y_val,
            validation_query=X_val,
            test_query=X_test,
            n_classes=len(outputs),
            node_index=node_index,
        )
        validation_node = _normalized_probabilities(
            prediction.validation_probabilities,
            n_rows=len(X_val),
            n_classes=len(outputs),
        )
        test_node = _normalized_probabilities(
            prediction.test_probabilities,
            n_rows=len(X_test),
            n_classes=len(outputs),
        )

        node_count += 1

        if is_leaf:
            return validation_node, test_node, node_classes

        validation_probabilities = np.zeros(
            (len(X_val), len(node_classes)),
            dtype=np.float64,
        )
        test_probabilities = np.zeros(
            (len(X_test), len(node_classes)),
            dtype=np.float64,
        )
        node_positions = {
            int(class_id): position
            for position, class_id in enumerate(node_classes)
        }
        for group_index, group_classes in enumerate(outputs):
            child_validation, child_test, child_classes = fit_predict_node(
                group_classes,
            )
            child_positions = [
                node_positions[int(class_id)] for class_id in child_classes
            ]
            validation_probabilities[:, child_positions] = (
                child_validation
                * validation_node[:, group_index : group_index + 1]
            )
            test_probabilities[:, child_positions] = (
                child_test * test_node[:, group_index : group_index + 1]
            )

        return (
            _normalized_probabilities(
                validation_probabilities,
                n_rows=len(X_val),
                n_classes=len(node_classes),
            ),
            _normalized_probabilities(
                test_probabilities,
                n_rows=len(X_test),
                n_classes=len(node_classes),
            ),
            node_classes,
        )

    validation_probabilities, test_probabilities, probability_classes = (
        fit_predict_node(ordered_classes)
    )
    validation_output = np.zeros(
        (len(X_val), len(classes)),
        dtype=np.float64,
    )
    test_output = np.zeros(
        (len(X_test), len(classes)),
        dtype=np.float64,
    )
    for column, class_index in enumerate(probability_classes):
        validation_output[:, int(class_index)] = validation_probabilities[
            :, column
        ]
        test_output[:, int(class_index)] = test_probabilities[:, column]

    return HierarchicalPrediction(
        validation_probabilities=_normalized_probabilities(
            validation_output,
            n_rows=len(X_val),
            n_classes=len(classes),
        ),
        test_probabilities=_normalized_probabilities(
            test_output,
            n_rows=len(X_test),
            n_classes=len(classes),
        ),
    )
