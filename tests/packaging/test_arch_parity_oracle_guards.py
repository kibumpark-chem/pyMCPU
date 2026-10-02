"""Self-tests for the guards inside ``scripts/arch_parity_dump.py``.

Why this file exists, specifically:

``arch_parity_dump.py`` is the oracle that decides whether a compiler or
compile-flag change altered the physics. During Phase 6 it produced three
wrong answers in a row, and every one was a *check that could not fail*:

1. ``step_stats()["mu_by_kind"]`` is bound as a ``py::list`` of dicts
   (``src/bindings/bindings.cpp:690``). The recorder tested
   ``isinstance(raw, dict)``, found a list, and stored ``{}``. The single most
   diagnostic counter was therefore absent from every record, and its absence
   read as agreement.
2. The same section carries ``ns`` wall-clock timings. The filter kept
   integers only, with a comment claiming that excluded the timings -- but
   ``ns`` *is* an integer, so only the float ``avg_ns`` was dropped. Raw
   nanoseconds entered the comparison and every run reported a divergence in
   that section regardless of the physics. That is worse than a missing check:
   it manufactures failures, and it masked a genuinely bit-identical result.
3. The cross-check that compares a build's *declared* FP-contraction setting
   against its observed FMA count first read
   ``build_info()["build"]["fp_contract"]``. That value lives under
   ``build_info()["fp"]``, so the guard found ``None`` and never fired -- a
   brand-new check that was inert on the day it was written.

The lesson is not "add more checks". Each of those *was* a check. It is that a
check which cannot fail is indistinguishable from a passing check, so each one
needs a test that deliberately trips it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / "scripts" / "arch_parity_dump.py"


def _load_oracle():
    """Import the script by path: ``scripts/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location("_arch_parity_dump", ORACLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def oracle():
    if not ORACLE.exists():
        pytest.skip(f"{ORACLE} not present")
    return _load_oracle()


# --------------------------------------------------------------------------
# Guard 1: mu_by_kind must be read in the shape the bindings actually produce.
# --------------------------------------------------------------------------


def test_reads_the_list_shape_the_bindings_emit(oracle):
    """The real shape: a list of dicts, each naming its own kind."""
    stats = {
        "mu_by_kind": [
            {"kind": "pivot", "eval_pair_nonzero": 11, "eval_pair_calls": 40},
            {"kind": "kic", "eval_pair_nonzero": 7, "eval_pair_calls": 21},
            {"kind": "sidechain", "eval_pair_nonzero": 3, "eval_pair_calls": 9},
        ]
    }
    got = oracle.extract_mu_by_kind(stats)
    assert set(got) == {"pivot", "kic", "sidechain"}
    assert got["pivot"]["eval_pair_nonzero"] == 11
    assert got["kic"]["eval_pair_calls"] == 21


def test_reads_the_dict_shape_too(oracle):
    """Accepted as well, so a binding change in either direction is tolerated."""
    stats = {"mu_by_kind": {"pivot": {"eval_pair_nonzero": 5}}}
    assert oracle.extract_mu_by_kind(stats) == {"pivot": {"eval_pair_nonzero": 5}}


def test_eval_pair_nonzero_is_actually_captured(oracle):
    """The counter this section exists for must survive extraction.

    This is the assertion that would have failed for the whole of Phase 6.
    """
    stats = {"mu_by_kind": [{"kind": "pivot", "eval_pair_nonzero": 1}]}
    got = oracle.extract_mu_by_kind(stats)
    assert "eval_pair_nonzero" in got["pivot"]


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(None, id="absent"),
        pytest.param([], id="empty-list"),
        pytest.param({}, id="empty-dict"),
        pytest.param("unexpected", id="wrong-type"),
        pytest.param([{"kind": "pivot"}], id="entries-carry-no-integers"),
    ],
)
def test_refuses_to_record_an_empty_section(oracle, raw):
    """Silently storing ``{}`` is the failure this guard exists to prevent."""
    with pytest.raises(SystemExit):
        oracle.extract_mu_by_kind({"mu_by_kind": raw})


# --------------------------------------------------------------------------
# Guard 2: wall-clock must never enter the fingerprint.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("timing_key", ["ns", "avg_ns", "delta_ns", "total_ns"])
def test_wall_clock_is_excluded_by_name_not_by_type(oracle, timing_key):
    """``ns`` is an ``int``, so an int-only filter does not exclude it.

    A timing field in the fingerprint makes every comparison fail, including
    one between two genuinely identical builds.
    """
    stats = {"mu_by_kind": [{"kind": "pivot", timing_key: 123456, "n_steps": 4}]}
    got = oracle.extract_mu_by_kind(stats)
    assert timing_key not in got["pivot"], (
        f"{timing_key} reached the fingerprint; two runs of the SAME build "
        f"would then report a spurious divergence"
    )
    assert got["pivot"]["n_steps"] == 4, "non-timing counters must survive"


def test_two_runs_differing_only_in_timing_compare_equal(oracle):
    """The property that matters, stated directly."""
    a = oracle.extract_mu_by_kind(
        {"mu_by_kind": [{"kind": "pivot", "ns": 1_000, "eval_pair_nonzero": 9}]}
    )
    b = oracle.extract_mu_by_kind(
        {"mu_by_kind": [{"kind": "pivot", "ns": 2_000_000, "eval_pair_nonzero": 9}]}
    )
    assert a == b


def test_booleans_are_not_recorded_as_counters(oracle):
    """``bool`` is a subclass of ``int``; flags are not counters."""
    stats = {"mu_by_kind": [{"kind": "pivot", "used": True, "n_steps": 2}]}
    assert "used" not in oracle.extract_mu_by_kind(stats)["pivot"]


# --------------------------------------------------------------------------
# Guard 3: a declared compile flag must be checked against observed codegen.
# --------------------------------------------------------------------------


def _record(
    *,
    sha: str,
    fma: int,
    fp_contract: str | None,
    cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    build_info: dict[str, Any] = {"arch": {"tier": "v3"}}
    if fp_contract is not None:
        build_info["fp"] = {"fp_contract": fp_contract}
    return {
        "format_version": getattr(_record, "_version", 1),
        "build": {
            "mcpu_core_file": f"/tmp/{sha}/mcpu_core.so",
            "so_sha256": sha,
            "isa_counts": {"zmm": 0, "kmask": 0, "ymm": 100, "fma": fma},
            "build_info": build_info,
        },
        "cases": cases if cases is not None else [_case()],
    }


def _case(label: str = "chignolin-s1337", **overrides: Any) -> dict[str, Any]:
    case = {
        "label": label,
        "steps": 10,
        "n_atoms": 77,
        "n_residues": 10,
        "n_accepts": 4,
        "accept_bits": "0101010101",
        "coords_sha256": "abc",
        "energy_before": {"1": "0x1.0p+0"},
        "energy_after": {"1": "0x1.8p+0"},
        "raw_total_before": "0x1.0p+0",
        "raw_total_after": "0x1.8p+0",
        "weighted_before": {"1": "0x1.999999999999ap-2"},
        "weighted_after": {"1": "0x1.3333333333333p-1"},
        "weighted_total_before": "0x1.999999999999ap-2",
        "weighted_total_after": "0x1.3333333333333p-1",
        "energy_weights": {"1": float(0.4).hex(), "4": float(2.7).hex()},
        "move_stats": {"num_accept_pivot": 2},
        "proxy_stats": {"neighbor_num_cell_visits": 7},
        "step_ints": {"n_steps": 10},
        "mu_by_kind": {"pivot": {"eval_pair_nonzero": 3}},
        "mu_backend": "opencell_mu_BBO_SC",
        "hbond_backend": "opencell_typed_OH_grids",
    }
    case.update(overrides)
    return case


@pytest.fixture(autouse=True)
def _pin_format_version(oracle):
    _record._version = oracle.FORMAT_VERSION


def test_warns_when_a_declared_flag_did_not_reach_the_compiler(oracle, capsys):
    """The exact situation that produced a wrong answer.

    A build configured with ``MCPU_FP_CONTRACT=off`` whose CMake guard dropped
    the flag emits the same FMA count as the default build. Comparing the two
    yields "no difference", which reads as "the flag changes nothing" when in
    fact nothing was changed.
    """
    ref = _record(sha="aaaa", fma=1281, fp_contract="fast")
    cur = _record(sha="bbbb", fma=1281, fp_contract="off")
    oracle._compare(ref, cur)
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "not applied" in out


def test_notes_when_fma_differs_for_a_reason_other_than_this_flag(oracle, capsys):
    """Same declared setting, different codegen -- the compiler dimension."""
    ref = _record(sha="aaaa", fma=14, fp_contract="off")
    cur = _record(sha="bbbb", fma=22, fp_contract="off")
    oracle._compare(ref, cur)
    out = capsys.readouterr().out
    assert "NOTE" in out
    assert "other than this flag" in out


def test_no_false_alarm_when_flag_and_codegen_agree(oracle, capsys):
    ref = _record(sha="aaaa", fma=1281, fp_contract="fast")
    cur = _record(sha="bbbb", fma=22, fp_contract="off")
    oracle._compare(ref, cur)
    out = capsys.readouterr().out
    assert "WARNING" not in out
    assert "other than this flag" not in out


def test_guard_is_silent_when_a_build_predates_build_info(oracle, capsys):
    """An older record has no ``fp`` section; that must not crash or warn."""
    ref = _record(sha="aaaa", fma=1597, fp_contract=None)
    cur = _record(sha="bbbb", fma=1281, fp_contract=None)
    oracle._compare(ref, cur)
    assert "WARNING" not in capsys.readouterr().out


# --------------------------------------------------------------------------
# The verdict itself: identical must mean identical, and divergence must be
# both detected and localized to an MC step.
# --------------------------------------------------------------------------


def test_identical_records_pass(oracle, capsys):
    ref = _record(sha="aaaa", fma=22, fp_contract="off")
    cur = _record(sha="bbbb", fma=22, fp_contract="off")
    assert oracle._compare(ref, cur) == 0
    out = capsys.readouterr().out
    assert "IDENTICAL" in out
    assert "PASS" in out


def test_a_single_flipped_accept_bit_is_detected_and_localized(oracle, capsys):
    ref = _record(sha="aaaa", fma=22, fp_contract="off")
    cur = _record(
        sha="bbbb",
        fma=22,
        fp_contract="off",
        cases=[_case(accept_bits="0101011101")],
    )
    assert oracle._compare(ref, cur) == 1
    out = capsys.readouterr().out
    assert "DIVERGED in accept_bits" in out
    assert "first divergence at MC step 6" in out


def test_same_digest_comparison_is_flagged_as_a_null_test(oracle, capsys):
    """Comparing a build against itself proved nothing, and once read as PASS."""
    rec = _record(sha="same", fma=22, fp_contract="off")
    oracle._compare(rec, dict(rec))
    out = capsys.readouterr().out
    assert "SAME .so digest" in out


def test_format_version_mismatch_is_a_harness_error_not_a_divergence(oracle, capsys):
    ref = _record(sha="aaaa", fma=22, fp_contract="off")
    cur = _record(sha="bbbb", fma=22, fp_contract="off")
    cur["format_version"] = oracle.FORMAT_VERSION + 1
    assert oracle._compare(ref, cur) == 2


# --------------------------------------------------------------------------
# Guard 4: a verdict of PASS must be impossible when nothing was compared.
# --------------------------------------------------------------------------


def test_zero_compared_cases_is_not_a_pass(oracle, capsys):
    """The worst possible output: PASS having compared nothing.

    Every SKIP path bypassed both the `failures` and `harness_errors`
    counters, so a comparison in which all four cases skipped -- a reference
    recorded with a different case set, or a missing input PDB turning every
    case into ``{"skipped": ...}`` via ``_spawn`` -- printed
    ``PASS: every case bit-identical across two different builds`` and exited
    0. Coverage is now part of the verdict.
    """
    ref = _record(sha="aaaa", fma=22, fp_contract="off",
                  cases=[_case(label="chignolin-s1337")])
    cur = _record(sha="bbbb", fma=22, fp_contract="off",
                  cases=[_case(label="a-case-the-reference-never-had")])
    assert oracle._compare(ref, cur) == 2
    out = capsys.readouterr().out
    assert "zero cases were actually compared" in out
    assert "PASS: every case" not in out


def test_all_cases_skipped_is_not_a_pass(oracle, capsys):
    ref = _record(sha="aaaa", fma=22, fp_contract="off")
    cur = _record(sha="bbbb", fma=22, fp_contract="off",
                  cases=[_case(skipped="missing pymcpu/data/1uao.pdb")])
    assert oracle._compare(ref, cur) == 2
    assert "PASS: every case" not in capsys.readouterr().out


def test_coverage_is_reported_on_a_real_comparison(oracle, capsys):
    ref = _record(sha="aaaa", fma=22, fp_contract="off")
    cur = _record(sha="bbbb", fma=22, fp_contract="off")
    assert oracle._compare(ref, cur) == 0
    assert "compared 1 case(s); skipped 0" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Guard 5: the fingerprint must witness the per-group outer weights.
# --------------------------------------------------------------------------


def test_a_one_ulp_weight_difference_is_detected(oracle, capsys):
    """The guard that justifies recording weighted quantities at all.

    The fingerprint used to hold only ``energy_breakdown(weighted=False)``, so
    an outer weight reached it ONLY by flipping a Metropolis accept. A wrong
    weight is a *static* offset -- unlike a coordinate perturbation it does not
    amplify -- so its per-step flip probability is ~beta*dw*|dE_group|, about
    5e-08..5e-07 at one ULP of the Mu weight (2.98e-08) and T=0.6. Over the
    4,800 MC steps this harness samples that is ~1e-03, i.e. a one-ULP weight
    regression would have been reported as PASS ~999 times in 1000.

    With the weight vector in the fingerprint it is caught deterministically,
    which is what this test pins.
    """
    import numpy as np

    mu = np.float32(0.4)
    mu_plus_1ulp = np.nextafter(mu, np.float32(1.0))
    assert mu_plus_1ulp != mu

    ref = _record(sha="aaaa", fma=22, fp_contract="off")
    cur = _record(sha="bbbb", fma=22, fp_contract="off",
                  cases=[_case(energy_weights={"1": float(mu_plus_1ulp).hex()})])
    assert oracle._compare(ref, cur) == 1
    assert "DIVERGED in energy_weights" in capsys.readouterr().out


def test_weighted_energy_difference_is_detected(oracle, capsys):
    ref = _record(sha="aaaa", fma=22, fp_contract="off")
    cur = _record(sha="bbbb", fma=22, fp_contract="off",
                  cases=[_case(weighted_total_after=float(-1.5).hex())])
    assert oracle._compare(ref, cur) == 1
    assert "DIVERGED in weighted_total_after" in capsys.readouterr().out
