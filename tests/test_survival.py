"""Pin ``Stryke.survival`` to the original ``simulation`` methods in ``Stryke/stryke.py`` and check its own contracts.

The original methods are pulled out of the source with ``ast`` and run unbound (they use no instance state), so the test needs only NumPy and
does not import Stryke's spreadsheet, hydrology or plotting dependencies.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

from Stryke import survival

SOURCE = Path(__file__).resolve().parent.parent / "Stryke" / "stryke.py"


def _original(name):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "simulation":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    module = ast.Module(body=[item], type_ignores=[])
                    namespace = {"np": np}
                    exec(compile(module, str(SOURCE), "exec"), namespace)
                    return namespace[name]
    raise AssertionError(f"{name} not found in {SOURCE}")


KAPLAN = dict(H=30.0, RPM=100.0, D=5.0, Q=200.0, ada=0.8, N=4)
PROP = dict(H=25.0, RPM=120.0, D=3.0, Q=180.0, ada=0.7, N=3, Qopt=160.0)
FRANCIS = dict(H=100.0, RPM=100.0, D=10.0, Q=500.0, ada=0.8, N=4, Qper=1.1, iota=1.1, D1=5.0, D2=10.0, B=1.0)


@pytest.mark.parametrize("length", [0.1, 0.5, 1.0])
def test_kaplan_matches_the_original_at_the_same_strike_position(length):
    original = _original("Kaplan")
    params = dict(KAPLAN, _lambda=0.2)
    np.random.seed(7)
    expected = original(None, length, params)
    np.random.seed(7)
    rR = np.random.uniform(0.3, 1.0, 1)
    assert survival.kaplan_survival(length, rR=rR, **KAPLAN) == pytest.approx(expected)


@pytest.mark.parametrize("length", [0.1, 0.5, 1.0])
def test_propeller_and_francis_match_the_original(length):
    assert survival.propeller_survival(length, **PROP) == pytest.approx(_original("Propeller")(None, length, dict(PROP, Qper=0.8, _lambda=0.2)))
    assert survival.francis_survival(length, **FRANCIS) == pytest.approx(_original("Francis")(None, length, dict(FRANCIS, _lambda=0.2)))


def test_a_seeded_generator_is_reproducible_and_vectorised():
    lengths = np.linspace(0.1, 1.0, 50)
    a = survival.kaplan_survival(lengths, rng=np.random.default_rng(3), **KAPLAN)
    b = survival.kaplan_survival(lengths, rng=np.random.default_rng(3), **KAPLAN)
    assert a.shape == lengths.shape and np.array_equal(a, b)
    assert np.all((a >= 0) & (a <= 1))


def test_longer_fish_survive_less_and_the_expected_value_lies_between_the_extremes():
    lengths = np.array([0.1, 0.3, 0.6, 1.0])
    expected = survival.kaplan_survival_expected(lengths, **KAPLAN)
    assert np.all(np.diff(expected) < 0)                                       # strike probability grows with length
    low = survival.kaplan_survival(lengths, rR=1.0, **KAPLAN)
    high = survival.kaplan_survival(lengths, rR=0.3, **KAPLAN)
    assert np.all(expected <= np.maximum(low, high) + 1e-12) and np.all(expected >= np.minimum(low, high) - 1e-12)


def test_the_expected_value_matches_a_large_seeded_sample():
    rng = np.random.default_rng(11)
    sample = survival.kaplan_survival(np.full(200000, 0.5), rng=rng, **KAPLAN).mean()
    assert survival.kaplan_survival_expected(0.5, **KAPLAN) == pytest.approx(sample, abs=2e-3)


def test_impingement_blocks_wide_fish_and_kills_those_that_cannot_swim_away():
    lengths = np.array([0.2, 2.0, 2.0])
    s, blocked, screened = survival.impingement(lengths, 0.1, rack_spacing=0.15, u_crit=np.array([1.0, 0.5, 3.0]), intake_vel=1.0)
    assert blocked.tolist() == [False, True, True]
    assert s.tolist() == [1.0, 0.0, 1.0] and screened.tolist() == [0.0, 0.0, 1.0]


@pytest.mark.parametrize("call", [
    lambda: survival.kaplan_survival(0.5, **KAPLAN),                               # neither rR nor rng
    lambda: survival.kaplan_survival(-0.1, rR=0.5, **KAPLAN), lambda: survival.kaplan_survival(0.5, rR=0.0, **KAPLAN),
    lambda: survival.kaplan_survival(0.5, rR=1.5, **KAPLAN), lambda: survival.kaplan_survival(0.5, rR=0.5, **dict(KAPLAN, RPM=0.0)),
    lambda: survival.kaplan_survival_expected(0.5, nodes=1, **KAPLAN), lambda: survival.kaplan_survival(np.nan, rR=0.5, **KAPLAN),
    lambda: survival.impingement(1.0, 0.1, 0.15, -1.0, 1.0)])
def test_bad_inputs_raise(call):
    with pytest.raises(ValueError):
        call()
