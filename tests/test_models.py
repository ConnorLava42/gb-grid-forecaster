"""Every model follows the same small interface and returns sorted quantiles."""

import numpy as np
import pandas as pd
import pytest

from gridforecast.features.build import FEATURES, TARGET, build_origin_table
from gridforecast.models.base import QCOLS, sort_quantiles
from gridforecast.models.baseline import SeasonalNaive
from gridforecast.models.ensemble import Average
from gridforecast.models.gbm import LGBMQuantile, LoadedLGBM


@pytest.fixture
def origins(table):
    o = build_origin_table(table)
    return o[o[TARGET].notna()]


def _check(preds, n):
    assert list(preds.columns[:3]) == QCOLS
    assert len(preds) == n
    assert preds[QCOLS].notna().all().all()
    q = preds[QCOLS].to_numpy()
    assert (np.diff(q, axis=1) >= 0).all()  # q10 <= q50 <= q90


@pytest.mark.parametrize(
    "factory",
    [
        SeasonalNaive,
        lambda: LGBMQuantile(max_trees=60),
        lambda: LGBMQuantile(max_trees=60, residual=True),
        lambda: Average([SeasonalNaive, lambda: LGBMQuantile(max_trees=60)]),
    ],
)
def test_model_interface(origins, factory):
    X, y = origins[FEATURES], origins[TARGET]
    model = factory().fit(X, y)
    _check(model.predict(X.tail(200)), 200)


def test_sort_quantiles_fixes_crossing():
    crossed = pd.DataFrame({"q10": [5.0], "q50": [3.0], "q90": [4.0]})
    assert sort_quantiles(crossed).iloc[0].tolist() == [3.0, 4.0, 5.0]


def test_lightgbm_save_and_load_give_identical_forecasts(origins, tmp_path):
    X, y = origins[FEATURES], origins[TARGET]
    model = LGBMQuantile(max_trees=60).fit(X, y)
    model.save(tmp_path, {"version": "test"})
    loaded = LoadedLGBM(tmp_path)
    np.testing.assert_allclose(
        model.predict(X.tail(50)).to_numpy(), loaded.predict(X.tail(50)).to_numpy(), rtol=1e-6
    )


@pytest.mark.nn
def test_mlp_interface(origins):
    pytest.importorskip("torch")
    from gridforecast.models.nn import MLPQuantile

    X, y = origins[FEATURES], origins[TARGET]
    model = MLPQuantile(max_epochs=2).fit(X, y)
    _check(model.predict(X.tail(100)), 100)
