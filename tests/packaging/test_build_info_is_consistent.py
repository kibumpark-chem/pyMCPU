"""``build_info()`` must be internally consistent and truthful.

The design that prevents a recurrence is that ``build_info()`` has **two
independent sources**: ``arch.*`` comes from a CMake-generated header, and
``isa.*`` comes from the compiler's own predefined macros. They can only agree
if the ``-march`` flag genuinely reached the compiler. So the cross-checks below
are not tautologies -- they compare CMake's intent against the compiler's
behaviour.

This is also the regression test for a bug that bit during Phase 6: an A/B
comparison ran with ``cmake.args`` silently overriding
``-Ccmake.define.MCPU_ARCH``, so both "builds" were the same binary and the
parity oracle reported PASS. A tier/ISA disagreement is the cheap way to notice
that the knob did nothing.
"""

from __future__ import annotations

import pytest

from pymcpu import mcpu_core


@pytest.fixture(scope="module")
def info() -> dict:
    fn = getattr(mcpu_core, "build_info", None)
    if fn is None:
        pytest.skip("build_info() absent; extension predates it -- rebuild")
    return fn()


def test_top_level_sections_present(info) -> None:
    for section in ("arch", "isa", "compiler", "build", "fp", "features", "deps"):
        assert section in info, f"build_info() is missing the {section!r} section"
        assert isinstance(info[section], dict)


def test_arch_is_reported(info) -> None:
    arch = info["arch"]
    assert arch["tier"], "no arch tier reported"
    if arch["tier"] != "none":
        assert arch["march"], (
            f"tier {arch['tier']!r} reports no -march value; either the flag "
            f"was not applied or the generated header was not regenerated"
        )
        assert arch["march"] in arch["flags"]


def test_tier_and_compiler_macros_agree(info) -> None:
    """The load-bearing assertion: CMake's intent vs the compiler's macros.

    A disagreement here means ``-march`` did not reach the compiler -- which is
    invisible to every other check, because the build still succeeds and the
    tests still pass. It just silently produces a binary for the wrong CPU.
    """
    tier, isa = info["arch"]["tier"], info["isa"]

    if tier == "v4":
        assert isa["avx512f"], "tier v4 but __AVX512F__ is not defined"
        assert isa["avx512vl"], "tier v4 but __AVX512VL__ is not defined"
    if tier in {"v3", "v4"}:
        for feature in ("avx", "avx2", "fma", "bmi2"):
            assert isa[feature], f"tier {tier} but __{feature.upper()}__ is not defined"
    if tier == "v2":
        assert not isa["avx2"], "tier v2 must not define __AVX2__"
    if tier == "none":
        assert not isa["avx512f"], "tier none must not define __AVX512F__"


def test_fp_policy_is_safe(info) -> None:
    """Pins the configure log's "SAFE math only" claim, which nothing enforced.

    ``CMakeLists.txt`` prints that message unconditionally, and before
    ``build_info()`` there was no way to check it. ``-ffast-math`` in a Monte
    Carlo engine whose accept/reject decisions come from float32 threshold
    comparisons would be a correctness problem, not a speed tradeoff.
    """
    fp = info["fp"]
    assert fp["fast_math"] is False, "-ffast-math is enabled; see MuPotential.h:419"
    assert fp["associative_math"] is False, "FP reassociation is enabled"
    assert fp["reciprocal_math"] is False
    assert fp["finite_math_only"] is False
    assert fp["fp_contract"], "fp_contract should be a value or 'compiler-default'"


def test_no_value_is_an_empty_placeholder(info) -> None:
    """Catches a generated header that was templated but never substituted."""
    empties = [
        f"{section}.{key}"
        for section, payload in info.items()
        for key, value in payload.items()
        # arch.march and arch.flags are legitimately empty for tier `none`.
        if value == "" and not (section == "arch" and info["arch"]["tier"] == "none")
    ]
    assert not empties, f"build_info() has unsubstituted/empty values: {empties}"
    # An unsubstituted @VAR@ is the classic configure_file failure.
    leftovers = [
        f"{s}.{k}={v}"
        for s, payload in info.items()
        for k, v in payload.items()
        if isinstance(v, str) and v.startswith("@") and v.endswith("@")
    ]
    assert not leftovers, f"configure_file did not substitute: {leftovers}"


def test_feature_flags_are_real_macro_reads(info) -> None:
    assert isinstance(info["features"]["MCPU_USE_POOLED_PROPOSAL"], bool)


def test_release_builds_actually_get_lto(info) -> None:
    """A Release build must really have LTO, not merely have requested it.

    Regression test for a silent, build-path-dependent loss of link-time
    optimisation. ``check_ipo_supported()`` was called without ``LANGUAGES``,
    so it checked every enabled language -- and FetchContent'd Eigen calls
    ``enable_language(Fortran)`` from its blas/, lapack/ and test/
    subdirectories. CMake has no IPO support for Fortran, so the check
    returned false and ``INTERPROCEDURAL_OPTIMIZATION`` was never set.

    The damage was invisible because it depended on how Eigen was found: a
    local ``cmake`` build against a system Eigen kept LTO, while the wheel,
    CI and conda-forge builds -- which FetchContent Eigen -- lost it. Every
    pre-release benchmark was therefore taken on a build configuration no
    user received. Nothing errored and no test failed.
    """
    if info["build"].get("type") != "Release":
        pytest.skip(f"build type is {info['build'].get('type')!r}, not Release")
    assert info["build"]["lto"] is True, (
        "Release build reports lto=False. Check the configure log for "
        "'LTO/IPO not supported' -- if it names a language this project does "
        "not use, a dependency enabled it and check_ipo_supported() needs its "
        "LANGUAGES argument scoped to CXX."
    )


def test_jcc_pad_report_is_consistent(info) -> None:
    """``jcc_pad`` is the EFFECTIVE state, decided by CMake's probes.

    GCC 8 with LTO drops ``-Wa,...`` at the link without a warning, so a
    requested-but-lost padding must show up as ``jcc_pad=False`` with a
    reason, never as ``True``. A forced ``MCPU_JCC_PAD=ON`` that could not
    be honoured fails at configure time and never reaches this test.
    """
    build = info["build"]
    mode = build["jcc_pad_mode"].upper()
    assert isinstance(build["jcc_pad"], bool)
    assert mode in {"AUTO", "ON", "OFF", "TRUE", "FALSE", "YES", "NO", "1", "0"}, mode
    assert build["jcc_pad_reason"], "jcc_pad_reason must say why"
    if mode in {"OFF", "FALSE", "NO", "0"}:
        assert build["jcc_pad"] is False
    if mode in {"ON", "TRUE", "YES", "1"}:
        assert build["jcc_pad"] is True
