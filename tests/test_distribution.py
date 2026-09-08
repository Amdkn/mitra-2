"""Regression predictive-distribution interface.

Everything here runs on CPU: the mixture math is numpy, the decode surgery is
exercised on a stand-in of AutoGluon's stock ``predict`` loop, and the
collector on stub trainers. The GPU end-to-end path is covered by the card
example (``predict_distribution`` on california housing).
"""
import inspect

import numpy as np
import pytest
import torch

from mitra_finetune.distribution import RegressionDistribution

# Global read by the rewritten decode when this module's function is
# transformed by ``_mean_decode_source`` (its globals are this module's).
_mitra_finetune_distribution_sink = None


def _softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _random_members(rng, n_members=3, n_samples=5, n_bins=40):
    edges = np.stack(
        [
            np.linspace(rng.uniform(-3, -1), rng.uniform(1, 3), n_bins + 1)
            for _ in range(n_members)
        ]
    )
    logits = 2.0 * rng.normal(size=(n_members, n_samples, n_bins))
    return edges, _softmax(logits).astype(np.float32)


def _member(dist, m):
    return RegressionDistribution(dist.bin_edges[m], dist.probabilities[m])


def _numeric_crps(dist, y, n_grid=400_001):
    lo, hi = dist.support
    lo, hi = min(lo, y.min()) - 1.0, max(hi, y.max()) + 1.0
    x = np.linspace(lo, hi, n_grid)
    f = dist.cdf(np.broadcast_to(x, (dist.n_samples, n_grid)))
    h = (x[None, :] >= y[:, None]).astype(float)
    g = (f - h) ** 2
    return np.trapezoid(g, x, axis=1) if hasattr(np, "trapezoid") else np.trapz(g, x, axis=1)


class TestConstruction:
    def test_single_member_promotion(self):
        d = RegressionDistribution(np.array([0.0, 1.0, 2.0]), np.array([[0.5, 0.5]]))
        assert (d.n_members, d.n_samples, d.n_bins) == (1, 1, 2)
        assert d.bin_edges.shape == (1, 3) and d.probabilities.shape == (1, 1, 2)
        assert d.probabilities.dtype == np.float32
        assert "n_members=1" in repr(d)

    @pytest.mark.parametrize(
        "edges, probs, match",
        [
            (np.array([[0.0, 1.0, 2.0]]), np.ones((1, 2, 3)), "does not match"),
            (np.array([[0.0, 1.0, 1.0]]), np.ones((1, 2, 2)), "strictly increasing"),
            (np.array([[0.0, 1.0, 2.0]]), -np.ones((1, 2, 2)), "non-negative"),
            (np.array([[0.0, 1.0, 2.0]]), np.zeros((1, 2, 2)), "positive mass"),
        ],
    )
    def test_invalid_inputs_raise(self, edges, probs, match):
        with pytest.raises(ValueError, match=match):
            RegressionDistribution(edges, probs)

    def test_point_prediction_shape_checked(self):
        with pytest.raises(ValueError, match="point_prediction"):
            RegressionDistribution(
                np.array([[0.0, 1.0]]), np.ones((1, 3, 1)), point_prediction=np.zeros(2)
            )

    def test_query_shapes(self):
        rng = np.random.default_rng(0)
        d = RegressionDistribution(*_random_members(rng, n_samples=4))
        assert d.cdf(0.3).shape == (4,)
        assert d.cdf(np.zeros(4)).shape == (4,)
        assert d.cdf(np.zeros((4, 7))).shape == (4, 7)
        with pytest.raises(ValueError, match="values for 4 samples"):
            d.cdf(np.zeros(3))
        with pytest.raises(ValueError, match="one observation per sample"):
            d.crps(np.zeros((4, 2)))


class TestUniformMember:
    """One bin on [0, 1] is U(0, 1); every quantity has a closed form."""

    @pytest.fixture
    def uniform(self):
        return RegressionDistribution(np.array([0.0, 1.0]), np.ones((1, 3, 1)))

    def test_mean(self, uniform):
        np.testing.assert_allclose(uniform.mean, 0.5)

    def test_cdf_pdf_quantile(self, uniform):
        np.testing.assert_allclose(uniform.cdf(np.array([0.25, 0.5, 2.0])), [0.25, 0.5, 1.0])
        np.testing.assert_allclose(uniform.cdf(-1.0), 0.0)
        np.testing.assert_allclose(uniform.pdf(np.array([0.2, 1.0, 1.5])), [1.0, 1.0, 0.0])
        assert np.all(np.isneginf(uniform.log_prob(np.array([-0.1, 2.0, 5.0]))))
        np.testing.assert_allclose(uniform.quantile([0.0, 0.3, 1.0])[0], [0.0, 0.3, 1.0])

    def test_crps_closed_form(self, uniform):
        y = np.array([0.5, 0.1, -1.0])
        expected = np.array([0.5**3 / 3 + 0.5**3 / 3, 0.1**3 / 3 + 0.9**3 / 3, 1.0 + 1 / 3])
        np.testing.assert_allclose(uniform.crps(y), expected, rtol=1e-12)


class TestMixture:
    @pytest.fixture
    def dist(self):
        rng = np.random.default_rng(42)
        return RegressionDistribution(*_random_members(rng, n_members=4, n_samples=6))

    def test_cdf_is_member_average(self, dist):
        x = np.linspace(-4, 4, 50)
        xq = np.broadcast_to(x, (dist.n_samples, x.size))
        members = np.mean([_member(dist, m).cdf(xq) for m in range(dist.n_members)], axis=0)
        np.testing.assert_allclose(dist.cdf(xq), members, atol=1e-12)
        assert np.all(np.diff(dist.cdf(xq), axis=1) >= -1e-12)
        lo, hi = dist.support
        np.testing.assert_allclose(dist.cdf(lo), 0.0)
        np.testing.assert_allclose(dist.cdf(hi), 1.0)

    def test_mean_matches_definition(self, dist):
        p = dist.probabilities.astype(np.float64)
        p /= p.sum(axis=2, keepdims=True)
        centers = 0.5 * (dist.bin_edges[:, :-1] + dist.bin_edges[:, 1:])
        expected = np.einsum("mnb,mb->n", p, centers) / dist.n_members
        np.testing.assert_allclose(dist.mean, expected, rtol=1e-12)

    def test_pdf_integrates_to_one_and_matches_cdf_slope(self, dist):
        lo, hi = dist.support
        x = np.linspace(lo, hi, 200_001)
        xq = np.broadcast_to(x, (dist.n_samples, x.size))
        pdf = dist.pdf(xq)
        trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        np.testing.assert_allclose(trapz(pdf, x, axis=1), 1.0, rtol=1e-3)
        cdf = dist.cdf(xq)
        np.testing.assert_allclose(cdf[:, -1] - cdf[:, 0], 1.0, atol=1e-12)

    def test_quantile_inverts_cdf(self, dist):
        q = np.linspace(0.01, 0.99, 25)
        x = dist.quantile(q)
        assert x.shape == (dist.n_samples, q.size)
        np.testing.assert_allclose(dist.cdf(x), np.broadcast_to(q, x.shape), atol=1e-10)
        assert np.all(np.diff(x, axis=1) >= 0)
        assert dist.quantile(0.5).shape == (dist.n_samples,)

    def test_crps_matches_numeric_integration(self, dist):
        rng = np.random.default_rng(3)
        lo, hi = dist.support
        y = rng.uniform(lo - 0.5, hi + 0.5, size=dist.n_samples)
        y[0] = lo - 1.5  # below the support
        y[1] = hi + 0.7  # above the support
        y[2] = dist.bin_edges[1, 7]  # exactly on a knot
        np.testing.assert_allclose(dist.crps(y), _numeric_crps(dist, y), rtol=2e-3, atol=1e-5)

    def test_crps_is_zero_for_point_mass(self):
        d = RegressionDistribution(np.array([0.0, 1e-9]), np.ones((1, 2, 1)))
        np.testing.assert_allclose(d.crps(np.zeros(2)), 0.0, atol=1e-9)

    def test_knot_cdf_matches_pointwise_cdf(self, dist):
        rows = slice(0, dist.n_samples)
        knots = dist._knots
        direct = dist._cdf_block(rows, np.broadcast_to(knots, (dist.n_samples, knots.shape[0])))
        fast = dist._cdf_at_knots(rows)
        np.testing.assert_allclose(fast[:, 1:-1], direct[:, 1:-1], atol=1e-12)
        assert np.all(fast[:, 0] == 0.0) and np.all(fast[:, -1] == 1.0)

    def test_cdf_stays_within_unit_interval(self):
        # float32 softmax rows renormalized in float64 can sum to 1 + 2 ulp; the
        # cdf must still be bounded (downstream scorers assert F <= 1).
        rng = np.random.default_rng(0)
        rows = []
        for _ in range(1000):
            p = _softmax(2.0 * rng.normal(size=1000)).astype(np.float32)
            p64 = p.astype(np.float64)
            if np.cumsum(p64 / p64.sum())[-1] > 1.0:
                rows.append(p)
            if len(rows) == 3:
                break
        assert rows, "no row with a float64 total above one was drawn"
        d = RegressionDistribution(np.linspace(0.0, 1.0, 1001), np.stack(rows))
        x = np.concatenate([[np.nextafter(1.0, 0.0), 1.0, 1.5], np.linspace(0.0, 1.0, 2001)])
        f = d.cdf(np.broadcast_to(x, (d.n_samples, x.size)))
        assert f.min() >= 0.0 and f.max() <= 1.0
        np.testing.assert_array_equal(f[:, 1:3], 1.0)
        assert d._cdf_at_knots(slice(0, d.n_samples)).max() <= 1.0

    def test_non_finite_observations(self, dist):
        with pytest.raises(ValueError, match="NaN"):
            dist.cdf(np.full(dist.n_samples, np.nan))
        with pytest.raises(ValueError, match="NaN"):
            dist.crps(np.nan)
        with pytest.raises(ValueError, match="levels"):
            dist.quantile([0.1, np.nan])
        np.testing.assert_array_equal(dist.cdf(np.inf), 1.0)
        np.testing.assert_array_equal(dist.cdf(-np.inf), 0.0)
        assert np.all(np.isposinf(dist.crps(np.inf)))
        assert np.all(np.isneginf(dist.log_prob(np.inf)))

    def test_identity_equality_and_bounds(self, dist):
        twin = RegressionDistribution(dist.bin_edges.copy(), dist.probabilities.copy())
        assert dist == dist and dist != twin and len({dist, twin}) == 2
        with pytest.raises(ValueError, match="at least one member"):
            RegressionDistribution(np.empty((0, 5)), np.empty((0, 3, 4)))

    def test_mean_and_quantiles_are_shift_invariant(self):
        rng = np.random.default_rng(11)
        edges, probs = _random_members(rng, n_members=8, n_samples=20, n_bins=200)
        base = RegressionDistribution(edges, probs)
        shifted = RegressionDistribution(edges + 1e6, probs)
        np.testing.assert_allclose(shifted.mean - 1e6, base.mean, atol=1e-7)
        np.testing.assert_allclose(shifted.quantile([0.1, 0.9]) - 1e6, base.quantile([0.1, 0.9]), atol=1e-6)

    def test_arrays_are_private_and_read_only(self):
        rng = np.random.default_rng(12)
        edges, probs = _random_members(rng)
        point = rng.normal(size=probs.shape[1])
        point0 = point.copy()
        d = RegressionDistribution(edges, probs, point_prediction=point)
        support, q50, c = d.support, d.quantile(0.5), d.cdf(0.3)
        # the caller mutating its own buffers must not reach into the object
        edges += 10.0
        probs *= 0.5
        point += 1.0
        assert d.support == support
        np.testing.assert_array_equal(d.quantile(0.5), q50)
        np.testing.assert_array_equal(d.cdf(0.3), c)
        assert d.probabilities.sum(axis=2).min() > 0.99
        np.testing.assert_array_equal(d.point_prediction, point0)
        # and the object's arrays cannot be written through
        for arr in (d.bin_edges, d.probabilities, d.point_prediction):
            assert not arr.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                arr[...] = 0


# --- decode surgery on a stand-in of AutoGluon's stock predict loop ------------


class _StubPreprocessor:
    def __init__(self, y_min, y_max, mirrored):
        self.y_min, self.y_max = y_min, y_max
        self.random_mirror_regression = True
        self.regression_mirror = mirrored

    def inverse_transform_y(self, y):
        if self.regression_mirror:
            y = 1 - y
        return y * (self.y_max - self.y_min) + self.y_min


class _StubTrainer:
    def __init__(self, n_bins, y_min, y_max, mirrored):
        self.bins = torch.linspace(-0.5, 1.5, n_bins + 1)
        self.bin_width = self.bins[1] - self.bins[0]
        self.preprocessor = _StubPreprocessor(y_min, y_max, mirrored)


def _fake_stock_predict(self, x_support, y_support, x_query):
    y_pred_list = []
    for y_hat in x_query:
        y_hat = np.argmax(y_hat, axis=-1)
        y_hat = (self.bins[y_hat] + self.bin_width / 2).cpu().numpy()
        y_hat = self.preprocessor.inverse_transform_y(y_hat)
        y_pred_list.append(y_hat)
    return np.concatenate(y_pred_list, axis=0)


def _expected_mean_decode(trainer, chunks):
    centers = (trainer.bins[:-1] + trainer.bin_width / 2).numpy()
    out = []
    for logits in chunks:
        probs = _softmax(logits.astype(np.float32))
        out.append(trainer.preprocessor.inverse_transform_y((probs * centers).sum(-1)))
    return np.concatenate(out)


class TestDecodeSurgery:
    @pytest.mark.parametrize("mirrored", [False, True])
    def test_mean_decode_and_sink(self, mirrored, monkeypatch):
        from mitra_finetune.patches import _mean_decode_source

        decoded = _mean_decode_source(_fake_stock_predict, numpy_variant=True)
        rng = np.random.default_rng(0)
        chunks = [rng.normal(size=(5, 20)), rng.normal(size=(3, 20))]
        trainer = _StubTrainer(20, y_min=10.0, y_max=30.0, mirrored=mirrored)

        seen = []
        monkeypatch.setitem(
            globals(), "_mitra_finetune_distribution_sink", lambda tr, p: seen.append((tr, p))
        )
        out = decoded(trainer, None, None, chunks)
        np.testing.assert_allclose(out, _expected_mean_decode(trainer, chunks), rtol=1e-5)
        assert [p.shape for _, p in seen] == [(5, 20), (3, 20)]
        assert all(tr is trainer for tr, _ in seen)
        np.testing.assert_allclose(seen[0][1].sum(-1).numpy(), 1.0, rtol=1e-5)

        monkeypatch.setitem(globals(), "_mitra_finetune_distribution_sink", None)
        np.testing.assert_array_equal(decoded(trainer, None, None, chunks), out)

    def test_needle_mismatch_raises(self):
        from mitra_finetune.patches import _mean_decode_source

        def not_a_decode(self):
            return 1

        with pytest.raises(RuntimeError, match="bin-decode changed"):
            _mean_decode_source(not_a_decode, numpy_variant=True)


class TestCollector:
    def _run_member(self, collector, trainer, chunks):
        collector.begin_call(trainer, n_query=sum(len(c) for c in chunks))
        for logits in chunks:
            collector.on_chunk(trainer, torch.softmax(torch.as_tensor(logits).float(), dim=-1))
        y_pred = _expected_mean_decode(trainer, chunks)
        collector.end_call(trainer, y_pred)
        return y_pred

    def test_members_map_to_target_units(self):
        from mitra_finetune.patches import _RegressionDistributionCollector

        rng = np.random.default_rng(1)
        chunks = [rng.normal(size=(4, 50)), rng.normal(size=(2, 50))]
        plain = _StubTrainer(50, y_min=-2.0, y_max=6.0, mirrored=False)
        mirrored = _StubTrainer(50, y_min=100.0, y_max=104.0, mirrored=True)
        collector = _RegressionDistributionCollector()
        points = [self._run_member(collector, plain, chunks), self._run_member(collector, mirrored, chunks)]
        edges, probs, pts = collector.stacked()
        assert edges.shape == (2, 51) and probs.shape == (2, 6, 50) and pts.shape == (2, 6)
        np.testing.assert_allclose(pts, np.stack(points))
        # plain: linspace(-0.5, 1.5) scaled to [-2, 6] spans [-6, 10]
        np.testing.assert_allclose(edges[0, [0, -1]], [-6.0, 10.0], rtol=1e-6)
        # mirrored: 1 - e reverses the grid; edges stay ascending and span [98, 106]
        np.testing.assert_allclose(edges[1, [0, -1]], [98.0, 106.0], rtol=1e-6)
        assert np.all(np.diff(edges, axis=1) > 0)
        # the mirrored member's bins are reversed: bin 0 of the mapped grid is the
        # last normalized bin
        raw = _softmax(np.concatenate(chunks).astype(np.float32))
        np.testing.assert_allclose(probs[1], raw[:, ::-1], rtol=1e-6)
        np.testing.assert_allclose(probs[0], raw, rtol=1e-6)
        dist = RegressionDistribution(edges, probs)
        np.testing.assert_allclose(dist.mean, pts.mean(axis=0), rtol=1e-5)

    @pytest.mark.parametrize("mirrored", [False, True])
    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_large_offset_targets_pass_the_consistency_check(self, mirrored, dtype):
        # Unix timestamps over one day: |y_min| / range ~ 2e4. Unnormalized
        # float32 softmax rows would carry a y_min * (sum(p) - 1) error of
        # ~1e3 target units against a 1e-4 * span tolerance of ~17.
        from mitra_finetune.patches import _RegressionDistributionCollector

        rng = np.random.default_rng(3)
        chunks = [rng.normal(scale=3.0, size=(64, 1000)) for _ in range(4)]
        y_min, y_max = dtype(1.6e9), dtype(1.6e9 + 86400.0)
        trainer = _StubTrainer(1000, y_min=y_min, y_max=y_max, mirrored=mirrored)
        collector = _RegressionDistributionCollector()
        point = self._run_member(collector, trainer, chunks)  # must not raise
        edges, probs, pts = collector.stacked()
        np.testing.assert_allclose(pts[0], point)
        np.testing.assert_allclose(edges[0, [0, -1]], [1.6e9 - 43200.0, 1.6e9 + 129600.0])
        dist = RegressionDistribution(edges, probs)
        # float32 point predictions in target units carry ~1e-7 relative error
        atol = 1e-4 * 2 * 86400.0 + (1e-6 * 1.6e9 if point.dtype.itemsize < 8 else 0.0)
        np.testing.assert_allclose(dist.mean, point.astype(np.float64), atol=atol)

    @pytest.mark.parametrize("y_min, span", [(1e12, 1.0), (1e9, 1e-3)])
    def test_extreme_offset_float64_targets_pass_the_consistency_check(self, y_min, span):
        # |y_min| / range ~ 1e12: both the stock decode (m * range + y_min) and
        # the histogram mean carry float64 rounding proportional to |y_min|,
        # which exceeds 1e-4 * span; the tolerance floor must absorb it.
        from mitra_finetune.patches import _RegressionDistributionCollector

        rng = np.random.default_rng(5)
        chunks = [rng.normal(scale=3.0, size=(32, 1000)) for _ in range(2)]
        trainer = _StubTrainer(1000, y_min=np.float64(y_min), y_max=np.float64(y_min + span), mirrored=False)
        collector = _RegressionDistributionCollector()
        point = self._run_member(collector, trainer, chunks)  # must not raise
        edges, probs, pts = collector.stacked()
        np.testing.assert_allclose(pts[0], point)
        dist = RegressionDistribution(edges, probs)
        np.testing.assert_allclose(dist.mean, point, atol=1e-4 * 2 * span + 1e-12 * y_min)

    def test_inconsistent_point_predictions_raise(self):
        from mitra_finetune.patches import _RegressionDistributionCollector

        rng = np.random.default_rng(2)
        chunks = [rng.normal(size=(3, 30))]
        trainer = _StubTrainer(30, y_min=0.0, y_max=1.0, mirrored=True)
        collector = _RegressionDistributionCollector()
        collector.begin_call(trainer, n_query=3)
        collector.on_chunk(trainer, torch.softmax(torch.as_tensor(chunks[0]).float(), dim=-1))
        wrong = _expected_mean_decode(trainer, chunks) + 0.5
        with pytest.raises(RuntimeError, match="does not reproduce"):
            collector.end_call(trainer, wrong)

    def test_protocol_errors(self):
        from mitra_finetune.patches import _RegressionDistributionCollector

        trainer = _StubTrainer(10, 0.0, 1.0, False)
        other = _StubTrainer(10, 0.0, 1.0, False)
        collector = _RegressionDistributionCollector()
        with pytest.raises(RuntimeError, match="outside its predict call"):
            collector.on_chunk(trainer, torch.ones(1, 10) / 10)
        collector.begin_call(trainer, n_query=2)
        with pytest.raises(RuntimeError, match="outside its predict call"):
            collector.on_chunk(other, torch.ones(1, 10) / 10)
        collector.on_chunk(trainer, torch.ones(1, 10) / 10)
        with pytest.raises(RuntimeError, match="rows 1 != query rows 2"):
            collector.end_call(trainer, np.zeros(1))
        # abort discards the partial attempt (an outer OOM retry re-enters)
        collector.begin_call(trainer, n_query=2)
        collector.on_chunk(trainer, torch.ones(1, 10) / 10)
        collector.abort_call()
        assert collector.members == []
        with pytest.raises(RuntimeError, match="no bag member"):
            collector.stacked()


class TestPatchWiring:
    def test_reg_ce_predict_wrapper_and_capture_toggle(self):
        pytest.importorskip("autogluon.tabular.models.mitra")
        import autogluon.tabular.models.mitra._internal.core.trainer_finetune as tm

        from mitra_finetune.patches import (
            _REG_DIST_CAPTURE,
            capture_regression_distribution,
            install_reg_ce_patches,
        )

        install_reg_ce_patches(1000)
        install_reg_ce_patches(1000)  # idempotent
        assert tm.TrainerFinetune.predict.__name__ == "predict_with_distribution_capture"
        assert tm._mitra_finetune_distribution_sink is None
        # the rewritten decode of the stock predict references the sink
        decoded = tm.TrainerFinetune.predict.__closure__
        assert any(
            getattr(c.cell_contents, "__name__", "") == "predict" for c in decoded
        )
        with capture_regression_distribution() as collector:
            assert _REG_DIST_CAPTURE["collector"] is collector
            assert tm._mitra_finetune_distribution_sink == collector.on_chunk
            with pytest.raises(RuntimeError, match="not re-entrant"):
                with capture_regression_distribution():
                    pass
        assert _REG_DIST_CAPTURE["collector"] is None
        assert tm._mitra_finetune_distribution_sink is None


# --- API and runner wiring ---------------------------------------------------


def _pretrained_dir(tmp_path, name="ckpt"):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"stub")
    return d


class TestApi:
    def _model(self, tmp_path, monkeypatch, problem_type="regression", name="ckpt"):
        from mitra_finetune.api import MitraFinetune

        model = MitraFinetune(
            _pretrained_dir(tmp_path, name=name), problem_type=problem_type
        )
        monkeypatch.setattr(model, "_check_runtime", lambda: None)
        return model

    def test_predict_distribution_routes_through_run_view(self, tmp_path, monkeypatch):
        from mitra_finetune import api

        rng = np.random.default_rng(0)
        edges, probs = _random_members(rng, n_members=2, n_samples=3, n_bins=8)
        captured = {}

        def fake_run_view(view, *args, **kwargs):
            captured.update(kwargs)
            out = (np.arange(8.0), np.array([1.0, 2.0, 3.0]))
            if kwargs.get("return_distribution"):
                return (*out, {"bin_edges": edges, "probabilities": probs})
            return out

        monkeypatch.setattr(api, "run_view", fake_run_view)
        model = self._model(tmp_path, monkeypatch)
        model.fit(np.zeros((8, 2)), np.arange(8.0))
        dist = model.predict_distribution(np.zeros((3, 2)))
        assert captured["return_distribution"] is True
        assert isinstance(dist, RegressionDistribution)
        np.testing.assert_array_equal(dist.point_prediction, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(dist.bin_edges, edges)
        np.testing.assert_array_equal(model.val_proba_, np.arange(8.0))

        captured.clear()
        full = model.predict(np.zeros((3, 2)), output_type="full")
        assert isinstance(full, RegressionDistribution)
        captured.clear()
        point = model.predict(np.zeros((3, 2)))
        assert captured["return_distribution"] is False
        np.testing.assert_array_equal(point, [1.0, 2.0, 3.0])

    def test_guards(self, tmp_path, monkeypatch):
        model = self._model(tmp_path, monkeypatch)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict_distribution(np.zeros((3, 2)))
        with pytest.raises(ValueError, match="output_type"):
            model.predict(np.zeros((3, 2)), output_type="mode")
        cls = self._model(
            tmp_path, monkeypatch, problem_type="classification", name="cls_ckpt"
        )
        with pytest.raises(RuntimeError, match="regression-only"):
            cls.predict_distribution(np.zeros((3, 2)))
        with pytest.raises(RuntimeError, match="regression-only"):
            cls.predict(np.zeros((3, 2)), output_type="full")


class TestRunnerWiring:
    def test_capture_wraps_the_final_regression_predict(self):
        import mitra_finetune.runner as runner

        src = inspect.getsource(runner.child_main)
        # both re-predict blocks (official validation / heldout-in-support)
        assert src.count("with _distribution_scope(_want_dist) as _dist_capture:") == 2
        assert '_want_dist = bool(is_regression and payload.get("return_distribution"))' in src
        # saved after every predict pass, before the point predictions
        assert src.index("_save_regression_distribution(") < src.index('np.save(work / "test_proba.npy"')
        rv = inspect.getsource(runner.run_view)
        assert '"return_distribution": bool(return_distribution)' in rv
        assert 'work / "test_distribution.npz"' in rv

    def test_distribution_scope_off_is_noop(self):
        from mitra_finetune.runner import _distribution_scope

        with _distribution_scope(False) as collector:
            assert collector is None

    def test_reg_ce_install_precedes_predict_wrappers(self):
        import mitra_finetune.runner as runner

        src = inspect.getsource(runner.child_main)
        assert src.index("install_reg_ce_patches(n_bins)") < src.index("install_fast_predict_patch(")
        assert src.index("install_reg_ce_patches(n_bins)") < src.index("install_predict_support_cap_patch(")
