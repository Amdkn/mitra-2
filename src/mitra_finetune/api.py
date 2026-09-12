"""Public API: MitraFinetune."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .checkpoint import resolve_checkpoint
from .distribution import RegressionDistribution
from .feature_selection import select_features, take_columns, take_rows
from .hierarchy import NodePrediction, hierarchical_predict_proba
from .modes import BASELINE
from .runner import run_view


NATIVE_CLASS_LIMIT = 10


class MitraFinetune:
    """Fine-tune a pretrained Mitra checkpoint with the frozen recipe.

    The fit is one AutoGluon bagged fine-tuning run with
    probability-averaged child predictions. By default, eight folds produce
    eight fine-tuned child models, and validation comes from out-of-fold
    training predictions. Callers may instead provide one external
    validation set shared by all child trainers. Test labels are never
    read.

    Env flag ``MITRA_VAL_IN_SUPPORT=1`` (default off, uniform across tasks):
    the FINAL test-time predictions add the official validation rows to every
    bag child's in-context support, matching rival TabFMs that fold train+val
    into the context. Validation predictions themselves -- everything that
    drives selection (checkpointing and ``val_proba_`` consumed
    downstream) -- are still computed with train-only
    support BEFORE validation joins the support. Requires ``X_val``/``y_val``
    at ``fit``; a no-op otherwise. See ``runner.child_main`` for mechanics.

    Parameters
    ----------
    checkpoint_dir : str or Path
        Path to the pretrained Mitra classification checkpoint (a ``.pt``
        state dict, or a directory containing exactly one).
    time_limit : int, default 3600
        Wall-clock budget for the fit in seconds, shared by the configured
        bag children.
    eval_metric : str, optional
        AutoGluon eval metric for validation checkpointing and internal
        model selection (e.g. ``"roc_auc"``, ``"log_loss"``). Defaults to
        log loss. Matching this to your reporting metric matters: the
        study evaluated ROC-AUC tasks with ``eval_metric="roc_auc"``.
    device : str, default "cuda"
    random_state : int, default 0
    num_bag_folds : int, default 8
        Number of AutoGluon bag folds and fitted child models. Each child
        trains on ``(num_bag_folds - 1) / num_bag_folds`` of the training
        rows. Must be at least 2.

    Attributes
    ----------
    val_proba_ : ndarray
        Probabilities for the active validation source: out-of-fold
        training rows by default, or the external validation rows supplied
        to ``fit``.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        problem_type: str = "classification",
        time_limit: int = 3600,
        eval_metric: str | None = None,
        device: str = "cuda",
        random_state: int = 0,
        in_process: bool = False,
        num_bag_folds: int = 8,
    ) -> None:
        problem_type = problem_type.strip().lower()
        if problem_type not in ("classification", "regression"):
            raise ValueError(
                f"Unknown problem_type {problem_type!r}; expected "
                "'classification' or 'regression'"
            )
        self.problem_type = problem_type
        self.checkpoint = resolve_checkpoint(
            checkpoint_dir,
            task="REGRESSION" if problem_type == "regression" else "CLASSIFICATION",
        )
        self.native_class_limit_: int | None = (
            NATIVE_CLASS_LIMIT if problem_type == "classification" else None
        )
        self._view = BASELINE
        self.time_limit = int(time_limit)
        self.eval_metric = eval_metric
        self.device = device
        self.random_state = int(random_state)
        self.num_bag_folds = int(num_bag_folds)
        if self.num_bag_folds < 2:
            raise ValueError("num_bag_folds must be at least 2")
        self.in_process = bool(in_process)
        self.val_proba_: np.ndarray | None = None
        self._n_classes: int | None = None
        self._fit_data = None
        # Number of leading predict-query rows that play a VALIDATION role
        # (hierarchical nodes predict val+test as one combined table). Only
        # consumed when MITRA_VAL_IN_SUPPORT=1: those rows must keep
        # train-only ICL support so validation-based selection stays clean.
        self._val_query_head = 0

    @staticmethod
    def _check_runtime() -> None:
        try:
            import autogluon.tabular.models.mitra  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "AutoGluon with the Mitra extra is required: "
                'pip install "autogluon.tabular[mitra]>=1.6"'
            ) from error
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            import warnings

            warnings.warn(
                "flash-attn is not installed. Fine-tuning runs on torch's "
                "scaled_dot_product_attention either way; prediction uses "
                "flash-attn when it is available, which is faster and uses "
                "less memory at prediction shapes. "
                "Install with: pip install flash-attn --no-build-isolation",
                RuntimeWarning,
                stacklevel=3,
            )

    def fit(self, X, y, X_val=None, y_val=None) -> "MitraFinetune":
        """Register training data for the fit.

        If ``X_val`` and ``y_val`` are omitted, each bag child validates on
        its held-out fold, producing out-of-fold validation predictions.
        If both are supplied, every child still trains on its 7/8 bag
        subset but uses the shared external validation set for
        checkpointing.
        """
        self._check_runtime()
        if (X_val is None) != (y_val is None):
            raise ValueError(
                "X_val and y_val must either both be provided or both be None"
            )
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError(f"X and y have different lengths: {len(X)} != {len(y)}")
        if self.problem_type == "classification":
            classes = np.unique(y)
            self._n_classes = len(classes)
        if X_val is not None:
            y_val = np.asarray(y_val)
            if len(X_val) != len(y_val):
                raise ValueError(
                    "X_val and y_val have different lengths: "
                    f"{len(X_val)} != {len(y_val)}"
                )
            if self.problem_type == "classification":
                unknown_labels = set(np.unique(y_val)).difference(np.unique(y))
                if unknown_labels:
                    raise ValueError(
                        "y_val contains labels absent from y: "
                        f"{sorted(unknown_labels, key=str)}"
                    )
        if (
            self.problem_type == "classification"
            and self._n_classes > self.native_class_limit_
        ):
            if not np.array_equal(classes, np.arange(self._n_classes)):
                raise ValueError(
                    "Many-class labels must be contiguous integers starting "
                    "at zero"
                )
            if X_val is None:
                raise ValueError(
                    "Classification above the checkpoint's native "
                    f"{self.native_class_limit_}-class head requires X_val "
                    "and y_val"
                )
            validation_classes = np.unique(y_val)
            if not np.array_equal(validation_classes, classes):
                missing = set(classes).difference(validation_classes)
                raise ValueError(
                    "Many-class validation must contain every training "
                    f"class; missing {sorted(missing, key=str)}"
                )

        # Keep DataFrames intact: categorical/object dtypes matter for how
        # AutoGluon and Mitra preprocess features. Only y was coerced above.
        self._ksel_idx = None
        self._svd = None
        self._svd_cols = None
        X, X_val, self._ksel_idx, _svd_info = select_features(
            X, y, X_val,
            problem_type=self.problem_type,
            checkpoint=self.checkpoint,
            eval_metric=self.eval_metric,
            random_state=self.random_state,
            gate_view=self._view,
            time_limit=self.time_limit,
            native_class_limit=self.native_class_limit_ or NATIVE_CLASS_LIMIT,
            run_view_fn=run_view,
        )
        if _svd_info is not None:
            self._svd, self._svd_cols = _svd_info
        # Views are executed at predict time so test predictions come from
        # the same fitted bag as the out-of-fold validation predictions
        # (Mitra is an in-context learner; the fit retains the support set).
        self._fit_data = (X, y, X_val, y_val)
        return self

    def _execute_view(self, X_test, *, return_distribution: bool = False):
        """Run the bagged fit, returning (val, test) predictions.

        Each value is a probability matrix (classification) or a vector of
        continuous point predictions (regression). With
        ``return_distribution=True`` (regression) a third element carries the
        bag children's bin distributions behind the test predictions.
        """
        X_train, y_train, X_val, y_val = self._fit_data
        if getattr(self, "_ksel_idx", None) is not None:
            X_test = take_columns(X_test, self._ksel_idx)
        elif getattr(self, "_svd", None) is not None:
            import pandas as pd

            Xt_num = np.asarray(
                X_test.values if hasattr(X_test, "values") else X_test,
                dtype=float,
            )
            X_test = pd.DataFrame(
                self._svd.transform(Xt_num),
                columns=self._svd_cols,
                index=X_test.index if hasattr(X_test, "index") else None,
            )
        return run_view(
            self._view,
            self.checkpoint,
            X_train,
            y_train,
            X_test,
            time_limit=self.time_limit,
            device=self.device,
            X_val=X_val,
            y_val=y_val,
            eval_metric=self.eval_metric,
            seed=self.random_state,
            in_process=self.in_process,
            problem_type=self.problem_type,
            num_bag_folds=self.num_bag_folds,
            # VAL_IN_SUPPORT: leading X_test rows that are
            # validation-role queries (0 outside the hierarchy).
            # Ignored unless MITRA_VAL_IN_SUPPORT=1.
            val_query_rows=getattr(self, "_val_query_head", 0),
            return_distribution=return_distribution,
        )

    def _predict_native_proba(self, X_test) -> np.ndarray:
        val_proba, test_proba = self._execute_view(X_test)
        self.val_proba_ = val_proba
        return test_proba

    @staticmethod
    def _concat_rows(first, second):
        if hasattr(first, "iloc") or hasattr(second, "iloc"):
            import pandas as pd

            if not hasattr(first, "iloc"):
                first = pd.DataFrame(first, columns=second.columns)
            if not hasattr(second, "iloc"):
                second = pd.DataFrame(second, columns=first.columns)
            return pd.concat([first, second], axis=0, ignore_index=True)
        return np.concatenate(
            [np.asarray(first), np.asarray(second)],
            axis=0,
        )

    @staticmethod
    def _cleanup_runtime() -> None:
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _predict_hierarchical_proba(self, X_test) -> np.ndarray:
        X_train, y_train, X_val, y_val = self._fit_data
        if getattr(self, "_ksel_idx", None) is not None:
            X_test = take_columns(X_test, self._ksel_idx)

        def run_node(
            *,
            X_train,
            y_train,
            X_val,
            y_val,
            validation_query,
            test_query,
            n_classes,
            node_index,
            **_,
        ) -> NodePrediction:
            node_model = MitraFinetune(
                checkpoint_dir=self.checkpoint,
                problem_type="classification",
                time_limit=self.time_limit,
                eval_metric=self.eval_metric,
                device=self.device,
                random_state=self.random_state + node_index,
                in_process=self.in_process,
                num_bag_folds=self.num_bag_folds,
            )
            try:
                node_model.fit(
                    X_train,
                    y_train,
                    X_val=X_val,
                    y_val=y_val,
                )
                combined_query = self._concat_rows(
                    validation_query,
                    test_query,
                )
                # The combined pass predicts validation-role rows first.
                # Under MITRA_VAL_IN_SUPPORT=1 the runner keeps train-only
                # support for that head (its predictions feed val_proba_ and
                # any downstream val-based selection) and extends support
                # with the official validation set for the test tail only.
                # A no-op when the flag is off.
                node_model._val_query_head = len(validation_query)
                combined_probabilities = np.asarray(
                    node_model.predict_proba(combined_query),
                    dtype=np.float64,
                )
                split = len(validation_query)
                return NodePrediction(
                    validation_probabilities=combined_probabilities[:split],
                    test_probabilities=combined_probabilities[split:],
                )
            finally:
                del node_model
                self._cleanup_runtime()

        result = hierarchical_predict_proba(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            node_runner=run_node,
            max_classes=self.native_class_limit_,
            random_state=self.random_state,
        )
        self.val_proba_ = result.validation_probabilities
        return result.test_probabilities

    def predict_proba(self, X_test) -> np.ndarray:
        if self.problem_type != "classification":
            raise RuntimeError(
                "predict_proba is classification-only; call predict() for "
                "regression point predictions"
            )
        if self._fit_data is None:
            raise RuntimeError("fit() must be called before predict_proba()")
        if self._n_classes > self.native_class_limit_:
            return self._predict_hierarchical_proba(X_test)
        return self._predict_native_proba(X_test)

    def predict(self, X_test, output_type: str = "mean"):
        """Predict ``X_test``: class labels (classification) or targets (regression).

        Regression ``output_type``: ``"mean"`` (default) returns the point
        predictions, the mean of the predicted bin distribution; ``"full"``
        returns the distribution itself (see ``predict_distribution``).
        """
        if output_type not in ("mean", "full"):
            raise ValueError(
                f"Unknown output_type {output_type!r}; expected 'mean' or 'full'"
            )
        if self.problem_type == "regression":
            if output_type == "full":
                return self.predict_distribution(X_test)
            if self._fit_data is None:
                raise RuntimeError("fit() must be called before predict()")
            val_pred, test_pred = self._execute_view(X_test)
            self.val_proba_ = np.asarray(val_pred, dtype=np.float64)
            return np.asarray(test_pred, dtype=np.float64)
        if output_type != "mean":
            raise RuntimeError(
                "output_type='full' is regression-only; call predict_proba() "
                "for class probabilities"
            )
        return np.argmax(self.predict_proba(X_test), axis=1)

    def predict_distribution(self, X_test) -> RegressionDistribution:
        """Predict the full regression distribution of every row of ``X_test``.

        Runs the same bagged fine-tune as ``predict`` (one fit per call; read
        the point predictions from ``.point_prediction`` instead of calling
        ``predict`` again). Each bag child predicts a categorical distribution
        over the checkpoint's 1,000 target bins on its own grid in target
        units; the returned :class:`RegressionDistribution` holds every
        child's bin edges and probabilities and evaluates their equal-weight
        mixture exactly (``mean``, ``cdf``, ``pdf``/``log_prob``,
        ``quantile``, ``crps``). Memory: about ``4 * n_members * n_bins``
        bytes per test row (32 KB at 8 children). ``val_proba_`` is set as
        for ``predict``.

        By default the histograms are captured from the same forward pass that
        produces the point predictions, so ``mean`` reproduces
        ``point_prediction`` up to float32 rounding. With
        ``MITRA_HELDOUT_IN_SUPPORT=0`` (and ``MITRA_VAL_IN_SUPPORT`` off) they
        come from one additional forward-only pass over ``X_test``; above the
        predict-time support cap that pass draws its own support subsample, so
        ``mean`` can then differ from ``point_prediction`` (the gap is logged).
        """
        if self.problem_type != "regression":
            raise RuntimeError(
                "predict_distribution is regression-only; call predict_proba() "
                "for class probabilities"
            )
        if self._fit_data is None:
            raise RuntimeError("fit() must be called before predict_distribution()")
        val_pred, test_pred, distribution = self._execute_view(
            X_test, return_distribution=True
        )
        self.val_proba_ = np.asarray(val_pred, dtype=np.float64)
        return RegressionDistribution(
            bin_edges=distribution["bin_edges"],
            probabilities=distribution["probabilities"],
            point_prediction=np.asarray(test_pred, dtype=np.float64),
        )
