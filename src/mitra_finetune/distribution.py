"""Predictive distributions of the Mitra-v2 regressor.

The regression checkpoint predicts a categorical distribution over ``n_bins``
target bins per query row (regression as classification, see
``patches.install_reg_ce_patches``); ``MitraFinetune.predict`` reports its
mean. ``MitraFinetune.predict_distribution`` returns the distribution itself:
one histogram per bag child, each on its own grid in target units (the grid is
fixed in the child's normalized target space, so mapped back the grids can differ),
and the bagged predictive distribution is the equal-weight mixture of those
histograms. ``RegressionDistribution`` evaluates that mixture exactly, treating
each bin's mass as uniform within the bin.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Rows per block in the knot-based evaluations: bounds the (rows x knots)
# float64 temporaries to about 16 MB each (256 rows x 8,000 knots).
_ROW_BLOCK = 256


@dataclass(frozen=True, repr=False, eq=False)
class RegressionDistribution:
    """Equal-weight mixture of per-member histograms, one histogram per row.

    Parameters
    ----------
    bin_edges : ndarray of shape (n_members, n_bins + 1)
        Strictly increasing bin edges of every member, in target units. A 1-D
        array is treated as a single member.
    probabilities : ndarray of shape (n_members, n_samples, n_bins)
        Bin probabilities of every member for every sample (float32, rows
        renormalized on use). A 2-D array is treated as a single member.
    point_prediction : ndarray of shape (n_samples,), optional
        The point prediction of the same fit (what ``predict`` returns).
        ``mean`` reproduces it up to float32 rounding when both come from the
        same forward pass, the default in ``MitraFinetune``.

    Notes
    -----
    Every member is a piecewise-uniform density, so the mixture cdf is
    piecewise linear with knots at the union of all members' edges; ``cdf``,
    ``pdf``, ``quantile`` and ``crps`` are exact under that interpretation.
    """

    bin_edges: np.ndarray
    probabilities: np.ndarray
    point_prediction: np.ndarray | None = None
    _knots_cache: np.ndarray | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _knot_index_cache: tuple | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # Private read-only copies: the caches describe this grid forever.
        edges = np.array(self.bin_edges, dtype=np.float64, copy=True, order="C")
        probs = np.array(self.probabilities, dtype=np.float32, copy=True, order="C")
        if edges.ndim == 1:
            edges = edges[None, :]
        if probs.ndim == 2:
            probs = probs[None, :, :]
        if edges.ndim != 2 or probs.ndim != 3:
            raise ValueError(
                "bin_edges must be (n_members, n_bins + 1) and probabilities "
                "(n_members, n_samples, n_bins)"
            )
        n_members, n_edges = edges.shape
        if n_members == 0 or n_edges < 2:
            raise ValueError("at least one member with at least one bin is required")
        if probs.shape[0] != n_members or probs.shape[2] != n_edges - 1:
            raise ValueError(
                f"probabilities shape {probs.shape} does not match bin_edges "
                f"shape {edges.shape}"
            )
        if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges, axis=1) > 0):
            raise ValueError("bin_edges must be finite and strictly increasing")
        if not np.all(np.isfinite(probs)) or np.any(probs < 0):
            raise ValueError("probabilities must be finite and non-negative")
        if np.any(probs.sum(axis=2) <= 0):
            raise ValueError("every probability row must carry positive mass")
        edges.setflags(write=False)
        probs.setflags(write=False)
        object.__setattr__(self, "bin_edges", edges)
        object.__setattr__(self, "probabilities", probs)
        if self.point_prediction is not None:
            point = np.array(self.point_prediction, dtype=np.float64, copy=True)
            if point.shape != (probs.shape[1],):
                raise ValueError(
                    "point_prediction must have one value per sample; got "
                    f"shape {point.shape} for {probs.shape[1]} samples"
                )
            point.setflags(write=False)
            object.__setattr__(self, "point_prediction", point)

    @property
    def n_members(self) -> int:
        return self.bin_edges.shape[0]

    @property
    def n_samples(self) -> int:
        return self.probabilities.shape[1]

    @property
    def n_bins(self) -> int:
        return self.probabilities.shape[2]

    @property
    def support(self) -> tuple[float, float]:
        """Smallest and largest edge over all members: the mixture's range."""
        return float(self.bin_edges[:, 0].min()), float(self.bin_edges[:, -1].max())

    def __repr__(self) -> str:
        lo, hi = self.support
        return (
            f"RegressionDistribution(n_members={self.n_members}, "
            f"n_samples={self.n_samples}, n_bins={self.n_bins}, "
            f"support=[{lo:.4g}, {hi:.4g}])"
        )

    # -- internals ------------------------------------------------------------

    def _row_blocks(self, block: int = _ROW_BLOCK):
        for start in range(0, self.n_samples, block):
            yield slice(start, min(start + block, self.n_samples))

    def _probs64(self, rows) -> np.ndarray:
        """(n_members, n_rows, n_bins) float64 probabilities, rows renormalized."""
        p = self.probabilities[:, rows, :].astype(np.float64)
        p /= p.sum(axis=2, keepdims=True)
        return p

    @property
    def _knots(self) -> np.ndarray:
        """Sorted union of all members' edges: the mixture cdf is linear between them."""
        if self._knots_cache is None:
            object.__setattr__(self, "_knots_cache", np.unique(self.bin_edges.ravel()))
        return self._knots_cache

    def _query(self, y):
        """Broadcast ``y`` to (n_samples, n_queries); return it with the output kind."""
        y = np.asarray(y, dtype=np.float64)
        if np.isnan(y).any():
            raise ValueError("y contains NaN; drop or impute missing observations first")
        if y.ndim == 0:
            return np.broadcast_to(y, (self.n_samples, 1)), "scalar"
        if y.ndim == 1:
            if y.shape[0] != self.n_samples:
                raise ValueError(
                    f"y has {y.shape[0]} values for {self.n_samples} samples"
                )
            return y[:, None], "vector"
        if y.ndim == 2 and y.shape[0] == self.n_samples:
            return y, "matrix"
        raise ValueError(
            "y must be a scalar, one value per sample, or an (n_samples, k) array"
        )

    @staticmethod
    def _finish(out: np.ndarray, kind: str) -> np.ndarray:
        return out[:, 0] if kind in ("scalar", "vector") else out

    def _locate(self, member: int, yq: np.ndarray):
        """Bin index (clipped), inside/below/above masks, width and in-bin fraction of ``yq``."""
        e = self.bin_edges[member]
        j = np.searchsorted(e, yq, side="right") - 1
        below = j < 0
        above = j >= self.n_bins
        jc = np.clip(j, 0, self.n_bins - 1)
        inside = (yq >= e[0]) & (yq <= e[-1])
        width = e[jc + 1] - e[jc]
        frac = np.clip((yq - e[jc]) / width, 0.0, 1.0)
        return jc, inside, below, above, width, frac

    @property
    def _knot_index(self) -> tuple:
        """Per member, the location of every knot on that member's grid (cached)."""
        if self._knot_index_cache is None:
            knots = self._knots
            table = []
            for m in range(self.n_members):
                jc, inside, below, _, _, frac = self._locate(m, knots)
                table.append((jc, inside, below, frac))
            object.__setattr__(self, "_knot_index_cache", tuple(table))
        return self._knot_index_cache

    def _cdf_at_knots(self, rows) -> np.ndarray:
        """Mixture cdf of the rows at every knot, (n_rows, n_knots), pinned to 0 and 1 at the ends."""
        p = self._probs64(rows)
        n_rows = p.shape[1]
        out = np.zeros((n_rows, self._knots.shape[0]))
        for m, (jc, inside, below, frac) in enumerate(self._knot_index):
            cum0 = np.concatenate([np.zeros((n_rows, 1)), np.cumsum(p[m], axis=1)], axis=1)
            fm = cum0[:, jc] + p[m][:, jc] * frac
            out += np.where(inside, fm, np.where(below, 0.0, 1.0))
        out /= self.n_members
        # Renormalized float32 rows can sum to 1 + 2 ulp in float64; keep F in [0, 1].
        np.clip(out, 0.0, 1.0, out=out)
        out[:, 0] = 0.0
        out[:, -1] = 1.0
        return out

    def _cdf_block(self, rows, yq: np.ndarray) -> np.ndarray:
        """Mixture cdf at ``yq`` (n_rows, k) for the rows in ``rows``."""
        p = self._probs64(rows)
        n_rows = p.shape[1]
        cum0 = np.concatenate(
            [np.zeros((self.n_members, n_rows, 1)), np.cumsum(p, axis=2)], axis=2
        )
        out = np.zeros(yq.shape, dtype=np.float64)
        for m in range(self.n_members):
            jc, inside, below, above, _, frac = self._locate(m, yq)
            lo = np.take_along_axis(cum0[m], jc, axis=1)
            pj = np.take_along_axis(p[m], jc, axis=1)
            fm = np.where(inside, lo + pj * frac, np.where(below, 0.0, 1.0))
            out += fm
        return np.clip(out / self.n_members, 0.0, 1.0)

    def _pdf_block(self, rows, yq: np.ndarray) -> np.ndarray:
        p = self._probs64(rows)
        out = np.zeros(yq.shape, dtype=np.float64)
        for m in range(self.n_members):
            jc, inside, _, _, width, _ = self._locate(m, yq)
            pj = np.take_along_axis(p[m], jc, axis=1)
            out += np.where(inside, pj / width, 0.0)
        return out / self.n_members

    # -- public evaluation ------------------------------------------------------

    @property
    def mean(self) -> np.ndarray:
        """(n_samples,) mixture mean: the bagged point prediction."""
        # Work relative to the lowest edge so targets with a large common
        # offset (timestamps) keep their precision.
        shift = float(self.bin_edges[:, 0].min())
        centers = 0.5 * (self.bin_edges[:, :-1] + self.bin_edges[:, 1:]) - shift
        out = np.empty(self.n_samples)
        for rows in self._row_blocks():
            p = self._probs64(rows)
            out[rows] = np.einsum("mnb,mb->n", p, centers) / self.n_members + shift
        return out

    def cdf(self, y) -> np.ndarray:
        """P(Y <= y). ``y``: scalar, (n_samples,), or (n_samples, k)."""
        yq, kind = self._query(y)
        out = np.empty(yq.shape)
        for rows in self._row_blocks():
            out[rows] = self._cdf_block(rows, yq[rows])
        return self._finish(out, kind)

    def pdf(self, y) -> np.ndarray:
        """Mixture density at ``y`` (piecewise constant; 0 outside the support)."""
        yq, kind = self._query(y)
        out = np.empty(yq.shape)
        for rows in self._row_blocks():
            out[rows] = self._pdf_block(rows, yq[rows])
        return self._finish(out, kind)

    def log_prob(self, y) -> np.ndarray:
        """Log density at ``y``; ``-inf`` where the density is zero."""
        with np.errstate(divide="ignore"):
            return np.log(self.pdf(y))

    def quantile(self, q) -> np.ndarray:
        """Quantiles of every sample's mixture at levels ``q`` in [0, 1].

        Returns (n_samples,) for a scalar ``q`` and (n_samples, len(q)) for a
        1-D ``q``. Exact inverse of ``cdf`` (leftmost point where the cdf
        reaches ``q``).
        """
        q = np.asarray(q, dtype=np.float64)
        scalar = q.ndim == 0
        q1 = np.atleast_1d(q)
        if q1.ndim != 1 or not np.all((q1 >= 0) & (q1 <= 1)):
            raise ValueError("q must be a scalar or 1-D array of levels in [0, 1]")
        knots = self._knots
        n_knots = knots.shape[0]
        out = np.empty((self.n_samples, q1.shape[0]))
        for rows in self._row_blocks():
            n_rows = out[rows].shape[0]
            f = np.maximum.accumulate(self._cdf_at_knots(rows), axis=1)
            ar = np.arange(n_rows)
            for k, level in enumerate(q1):
                idx = np.minimum((f < level).sum(axis=1), n_knots - 1)
                i0 = np.maximum(idx - 1, 0)
                f0, f1 = f[ar, i0], f[ar, idx]
                k0, k1 = knots[i0], knots[idx]
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = np.where(f1 > f0, (level - f0) / (f1 - f0), 0.0)
                x = k0 + np.clip(t, 0.0, 1.0) * (k1 - k0)
                out[rows, k] = np.where(idx == 0, knots[0], x)
        return out[:, 0] if scalar else out

    def crps(self, y) -> np.ndarray:
        """Continuous ranked probability score of every sample against ``y``.

        ``y`` is a scalar or one observation per sample. Computed exactly:
        the mixture cdf is piecewise linear between the union of all members'
        edges (plus ``y``), so the integral of ``(F(x) - 1[x >= y])^2`` is a
        sum of closed-form quadratic pieces. Lower is better; same units as
        the target.
        """
        yq, kind = self._query(y)
        if kind == "matrix":
            raise ValueError("crps takes one observation per sample")
        yv = yq[:, 0]
        knots = self._knots
        n_knots = knots.shape[0]
        k0, k1 = knots[:-1], knots[1:]
        widths = k1 - k0
        out = np.empty(self.n_samples)
        for rows in self._row_blocks():
            yb = yv[rows]
            f = self._cdf_at_knots(rows)
            fy = self._cdf_block(rows, yb[:, None])[:, 0]
            # Intervals fully before y carry indicator 0, intervals starting
            # at or after y carry indicator 1; both make (F - 1[x >= y])
            # linear on the interval.
            ind = (k0[None, :] >= yb[:, None]).astype(np.float64)
            a = f[:, :-1] - ind
            b = f[:, 1:] - ind
            total = (widths[None, :] * (a * a + a * b + b * b) / 3.0).sum(axis=1)
            # The interval that contains y strictly inside is split at y.
            j = np.clip(np.searchsorted(knots, yb, side="right") - 1, 0, n_knots - 2)
            strict = (knots[j] < yb) & (yb < knots[j + 1])
            if strict.any():
                r = np.flatnonzero(strict)
                jr = j[r]
                a0, b0, fr = f[r, jr], f[r, jr + 1], fy[r]
                total[r] -= widths[jr] * (a0 * a0 + a0 * b0 + b0 * b0) / 3.0
                left = yb[r] - knots[jr]
                total[r] += left * (a0 * a0 + a0 * fr + fr * fr) / 3.0
                right = knots[jr + 1] - yb[r]
                a1, b1 = fr - 1.0, b0 - 1.0
                total[r] += right * (a1 * a1 + a1 * b1 + b1 * b1) / 3.0
            # Outside the support the cdf is 0 (left) or 1 (right).
            total += np.where(yb < knots[0], knots[0] - yb, 0.0)
            total += np.where(yb > knots[-1], yb - knots[-1], 0.0)
            out[rows] = total
        return out
