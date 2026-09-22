# Security policy

## Supported versions

pyMCPU is pre-1.0. Only the latest released version receives fixes.

| version | supported |
|---|---|
| 0.1.x | yes |
| < 0.1 | no |

## Reporting a vulnerability

Please report privately rather than opening a public issue:

- GitHub **Security → Report a vulnerability** on
  <https://github.com/kibumpark-chem/pyMCPU/security/advisories/new>, or
- email **kibum_park@fas.harvard.edu**.

Please include the version (`python -c "import pymcpu; print(pymcpu.__version__)"`),
your platform, and a reproducer. You can expect an acknowledgement within a
week; this is a research project maintained by one person, so please size your
expectations accordingly.

## Threat model — what is and is not in scope

pyMCPU is a scientific simulation engine. It is normally run on inputs the user
already trusts, so it is worth being explicit about where the boundaries are.

**In scope:**

- Memory-safety faults in the C++ core reachable from a *well-formed* PDB or
  parameter file — out-of-bounds reads/writes, use-after-free.
- A crafted **parameter archive** causing arbitrary code execution or writes
  outside the cache directory. Parameter sets are downloaded and unpacked, so
  path traversal during unpacking is a genuine concern.
- Checksum handling in the parameter-download path
  (`pymcpu/params.py`, `pymcpu/paramcodec.py`).

**Not in scope:**

- Crashes from malformed or adversarial **structure files**. Structures are
  parsed by [mdtraj](https://mdtraj.org); report those upstream. pyMCPU does
  not treat a PDB as untrusted input.
- Numerical disagreement between builds. This is expected and documented —
  see `docs/known_issues.md`. It is a
  reproducibility property, not a vulnerability.
- Resource exhaustion from a large system or a long run. That is the intended
  use.
- Anything requiring the attacker to already be able to run code as you.

## Notes for the cautious

- **Parameter downloads are checksum-verified.** `pymcpu/params.py` refuses to
  download without a recorded SHA-256. To avoid the network entirely, set
  `MCPU_NO_DOWNLOAD=1`, or point `MCPU_PARAMS_DIR` at a directory you control;
  a `pip`-installed wheel already carries its parameters and needs no network.
- **The compiled extension is built from source** on install unless you take a
  published wheel. `mcpu_core.build_info()` reports the compiler, the resolved
  `-march`, the LTO state and the floating-point flags of the binary you are
  actually running, so you can verify what you have.
- pyMCPU does not execute parameter-set content as code. Parameter files are
  numeric tables plus JSON/CSV metadata.
