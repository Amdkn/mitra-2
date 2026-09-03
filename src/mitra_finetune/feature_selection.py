"""High-dimensional feature selection for Mitra fine-tuning."""
from __future__ import annotations

import os
import warnings

import numpy as np


def select_features(X, y, X_val, *, problem_type, checkpoint, eval_metric,
                    random_state, gate_view, time_limit, native_class_limit,
                    run_view_fn):
    """Apply top-K feature selection if the table exceeds the feature cap.

    Wide tables push the in-context model off-distribution; scoring the
    columns and keeping only the strongest recovers it. A no-op when the
    feature count is at or below the cap (default 256, override with
    ``MITRA_REG_MAX_FEATURES`` / ``MITRA_CLS_MAX_FEATURES``). The kept index
    is returned so the same columns are taken from validation and test.

    Returns (X, X_val, ksel_idx, svd_info): ksel_idx is None if top-K FS was
    skipped; svd_info is None unless the SVD path ran, else a
    ``(fitted_pipeline, component_column_names)`` pair the caller must apply
    to the test table.
    """
    env_key = (
        "MITRA_REG_MAX_FEATURES"
        if problem_type == "regression"
        else "MITRA_CLS_MAX_FEATURES"
    )
    k = int(os.environ.get(env_key, "256"))
    if k <= 0 or X.shape[1] <= k:
        return X, X_val, None, None
    try:
        from sklearn.feature_selection import f_regression

        X_num = np.asarray(
            X.values if hasattr(X, "values") else X, dtype=float
        )
    except (TypeError, ValueError):
        # Non-numeric features (e.g. categoricals): leave the table as is.
        return X, X_val, None, None

    # Regression wide-table reduction: PCA to K components (the released
    # method for regression; on standardized inputs this is the truncated
    # SVD referenced in the report). Gated by MITRA_FS_METHOD, which
    # defaults to "svd" for regression and "select" (F-stat top-K) for
    # classification; set it explicitly to override either default.
    _fs_default = "svd" if problem_type == "regression" else "select"
    if os.environ.get("MITRA_FS_METHOD", _fs_default) == "svd":
        import pandas as pd
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        n_comp = int(min(k, X_num.shape[0], X_num.shape[1]))
        pipe = make_pipeline(
            SimpleImputer(strategy="mean"),
            StandardScaler(),
            PCA(n_components=n_comp, random_state=0),
        )
        Z = pipe.fit_transform(X_num)  # fit on train/support rows only
        svd_cols = [f"svd_{i}" for i in range(Z.shape[1])]
        warnings.warn(
            f"[fs-svd] {X_num.shape[1]} -> {Z.shape[1]} comps",
            RuntimeWarning,
            stacklevel=2,
        )
        Xn = pd.DataFrame(
            Z,
            columns=svd_cols,
            index=X.index if hasattr(X, "index") else None,
        )
        Xvn = None
        if X_val is not None:
            Xv_num = np.asarray(
                X_val.values if hasattr(X_val, "values") else X_val,
                dtype=float,
            )
            Xvn = pd.DataFrame(
                pipe.transform(Xv_num),
                columns=svd_cols,
                index=X_val.index if hasattr(X_val, "index") else None,
            )
        return Xn, Xvn, None, (pipe, svd_cols)

    fs_gate = os.environ.get("MITRA_FS_GATE", "dtype")
    native_gate_fallback = (
        problem_type == "classification"
        and fs_gate == "1"
        and len(np.unique(y)) > native_class_limit
    )
    if problem_type == "classification" and (
        fs_gate == "dtype" or native_gate_fallback
    ):
        # F-stat top-K denoises wide continuous-descriptor tables but loses
        # signal on diffuse binary/low-cardinality ones: skip FS unless the
        # columns are predominantly continuous. The validation gate uses
        # native Mitra fits, so hierarchy-sized targets fall back to this
        # non-model gate instead of exceeding the 10-class head.
        n_cont = sum(
            len(np.unique(X_num[:, j])) > 2 for j in range(X_num.shape[1])
        )
        min_cont = float(os.environ.get("MITRA_FS_DTYPE_MIN_CONT", "0.5"))
        keep = n_cont >= X_num.shape[1] * min_cont
        gate_name = (
            "fs-gate-hierarchy-fallback"
            if native_gate_fallback
            else "fs-dtype-gate"
        )
        warnings.warn(
            f"[{gate_name}] cont={n_cont}/{X_num.shape[1]} -> "
            f"{'KEEP FS' if keep else 'DROP FS'}",
            RuntimeWarning,
            stacklevel=2,
        )
        if not keep:
            return X, X_val, None, None

    with np.errstate(divide="ignore", invalid="ignore"):
        if problem_type == "regression":
            scores, _ = f_regression(X_num, np.asarray(y, dtype=float), center=True)
        else:
            _scorer = os.environ.get("MITRA_CLS_FS_SCORER", "f_classif")
            _yv = np.asarray(y)
            if _scorer == "mutual_info":
                from sklearn.feature_selection import mutual_info_classif
                _disc = np.array([len(np.unique(X_num[:, _j])) <= 2
                                  for _j in range(X_num.shape[1])])
                scores = mutual_info_classif(
                    X_num, _yv, discrete_features=_disc, random_state=0)
            elif _scorer == "auc":
                from sklearn.metrics import roc_auc_score as _auc
                _cls = np.unique(_yv)
                _targets = _cls[1:] if len(_cls) == 2 else _cls
                scores = np.zeros(X_num.shape[1])
                for _j in range(X_num.shape[1]):
                    _b = 0.0
                    for _c in _targets:
                        try:
                            _b = max(_b, abs(_auc(
                                (_yv == _c).astype(int), X_num[:, _j]) - 0.5))
                        except Exception:
                            pass
                    scores[_j] = _b
            else:
                from sklearn.feature_selection import f_classif
                scores, _ = f_classif(X_num, _yv)

    # Match the evaluation harness exactly: rank by F-statistic (no sort of
    # the surviving indices), non-finite scores sink to the bottom.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    cand_idx = np.argsort(scores)[::-1][:k]

    if (
        problem_type == "classification"
        and fs_gate == "1"
        and not native_gate_fallback
        and not _fs_gate_keeps_fs(
            X, y, cand_idx,
            checkpoint=checkpoint,
            eval_metric=eval_metric,
            random_state=random_state,
            gate_view=gate_view,
            time_limit=time_limit,
            run_view_fn=run_view_fn,
        )
    ):
        # Val-gate says full-width beats FS on this table: keep it whole.
        return X, X_val, None, None

    return (
        _take_columns(X, cand_idx),
        None if X_val is None else _take_columns(X_val, cand_idx),
        cand_idx,
        None,
    )


def _fs_gate_keeps_fs(X, y, idx, *, checkpoint, eval_metric, random_state,
                      gate_view, time_limit, run_view_fn):
    """Per-table val-gate: keep FS only if it beats full-width on a holdout.

    Fits full-width AND FS-selected columns at FULL deployment steps on one
    stratified 80/20 holdout and keeps FS only when it scores strictly
    better. Cheap proxies were empirically unreliable: on the
    diffuse-binary hiva table FS looks better zero-shot yet
    hurts after fine-tuning, and the full-vs-FS holdout margin is noise
    until full steps. The decision is deterministic per (checkpoint, table)
    so it is cached to disk (MITRA_FS_GATE_CACHE).
    """
    import json

    yv = np.asarray(y)
    metric = eval_metric or (
        "roc_auc" if len(np.unique(yv)) == 2 else "log_loss"
    )
    scorer = os.environ.get("MITRA_CLS_FS_SCORER", "f_classif")
    key = "|".join(
        str(t)
        for t in (
            os.path.basename(str(checkpoint).rstrip("/")),
            X.shape[0],
            X.shape[1],
            len(idx),
            metric,
            scorer,
        )
    )
    cache_path = os.environ.get(
        "MITRA_FS_GATE_CACHE", "/tmp/mitra_fs_gate_cache.json"
    )
    try:
        cache = json.load(open(cache_path))
    except Exception:
        cache = {}
    if key in cache:
        return bool(cache[key])

    cap = int(os.environ.get("MITRA_FS_GATE_CAP", "256"))
    from sklearn.model_selection import train_test_split

    tr, ho = train_test_split(
        np.arange(len(yv)),
        test_size=0.2,
        random_state=random_state,
        stratify=yv,
    )
    X_tr, X_ho = _take_rows(X, tr), _take_rows(X, ho)
    y_tr, y_ho = yv[tr], yv[ho]
    score_full = _gate_fit_score(
        X_tr, y_tr, X_ho, y_ho, None, yv, metric, cap,
        checkpoint=checkpoint, eval_metric=eval_metric,
        random_state=random_state, gate_view=gate_view,
        time_limit=time_limit, run_view_fn=run_view_fn,
    )
    score_fs = _gate_fit_score(
        X_tr, y_tr, X_ho, y_ho, idx, yv, metric, cap,
        checkpoint=checkpoint, eval_metric=eval_metric,
        random_state=random_state, gate_view=gate_view,
        time_limit=time_limit, run_view_fn=run_view_fn,
    )
    keep = bool(score_fs > score_full)
    warnings.warn(
        f"[fs-gate] full={score_full:.5f} fs={score_fs:.5f} -> "
        f"{'KEEP FS' if keep else 'DROP FS'} ({key})",
        RuntimeWarning,
        stacklevel=2,
    )
    cache[key] = keep
    try:
        json.dump(cache, open(cache_path, "w"))
    except Exception:
        pass
    return keep


def _gate_fit_score(X_tr, y_tr, X_ho, y_ho, idx, y_all, metric, cap,
                    *, checkpoint, eval_metric, random_state, gate_view,
                    time_limit, run_view_fn):
    """One gate fit: full-step bagged fit on the 80% train, scored on the
    20% holdout (higher is better). ``idx=None`` fits the full width."""
    def take(A):
        return A if idx is None else _take_columns(A, idx)

    _val, test_o = run_view_fn(
        gate_view,
        checkpoint,
        take(X_tr),
        y_tr,
        take(X_ho),
        time_limit=time_limit,
        device="cuda",
        eval_metric=eval_metric,
        in_process=False,
        seed=random_state,
        problem_type="classification",
        num_bag_folds=2,
        gate_cap=cap,
    )
    classes = sorted(np.unique(np.asarray(y_all)).tolist())
    pos = {c: i for i, c in enumerate(classes)}
    y_enc = np.asarray([pos[v] for v in y_ho])
    from sklearn.metrics import log_loss, roc_auc_score

    if metric == "roc_auc" and len(classes) == 2:
        return float(roc_auc_score(y_enc, test_o[:, 1]))
    return -float(
        log_loss(y_enc, test_o, labels=list(range(len(classes))))
    )


def take_columns(X, idx):
    """Column-subset a DataFrame (by position) or a 2-D array alike."""
    return _take_columns(X, idx)


def take_rows(X, idx):
    """Row-subset a DataFrame (by position) or a 2-D array alike."""
    return _take_rows(X, idx)


def _take_columns(X, idx):
    return X.iloc[:, idx] if hasattr(X, "iloc") else np.asarray(X)[:, idx]


def _take_rows(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else np.asarray(X)[idx]
