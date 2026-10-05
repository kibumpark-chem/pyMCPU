cmake_minimum_required(VERSION 3.15)
project(pymcpu CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
include(FetchContent)

# Look for dependencies in the environment BEFORE reaching for the network.
#
# This matters beyond politeness: conda-forge builds run with no network
# access, so a FetchContent_MakeAvailable(eigen) that is not guarded by a
# find_package cannot configure there at all -- the recipe fails before it
# compiles a line. Eigen was previously fetched unconditionally.
#
# MCPU_FETCH_DEPS=OFF turns a missing dependency into a hard configure error
# instead of a silent download, which is what a packaging build wants.
option(MCPU_FETCH_DEPS
    "Download missing dependencies with FetchContent (OFF = hard error)" ON)

# NO_CMAKE_PACKAGE_REGISTRY is load-bearing, not tidiness. CMake's per-user
# package registry (~/.cmake/packages/Eigen3) accumulates an entry for every
# build tree that ever ran export(PACKAGE Eigen3) -- FetchContent'd Eigen does
# exactly that. On this machine it held SIX entries pointing into stale pyMCPU
# build directories, and find_package silently selected one, producing
# "Imported target Eigen3::Eigen includes non-existent path" at configure time.
# A build must not depend on a machine-local side channel it cannot see.
find_package(Eigen3 3.4 QUIET NO_MODULE
             NO_CMAKE_PACKAGE_REGISTRY
             NO_CMAKE_SYSTEM_PACKAGE_REGISTRY)
if(Eigen3_FOUND)
    message(STATUS "Eigen3 ${Eigen3_VERSION}: found in the environment")
elseif(MCPU_FETCH_DEPS)
    message(STATUS "Eigen3 >=3.4 not found — fetching via FetchContent")
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
    )
    FetchContent_MakeAvailable(eigen)
else()
    message(FATAL_ERROR
        "Eigen3 >=3.4 not found and MCPU_FETCH_DEPS=OFF. Install Eigen "
        "(conda: `eigen`; Debian: `libeigen3-dev`) or set CMAKE_PREFIX_PATH.")
endif()

find_package(pybind11 2.12 QUIET)
if(pybind11_FOUND)
    message(STATUS "pybind11 ${pybind11_VERSION}: found in the environment")
elseif(MCPU_FETCH_DEPS)
    message(STATUS "pybind11 >=2.12 not found — fetching via FetchContent")
    set(PYBIND11_FINDPYTHON ON)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
else()
    message(FATAL_ERROR
        "pybind11 >=2.12 not found and MCPU_FETCH_DEPS=OFF. Install pybind11 "
        "or set CMAKE_PREFIX_PATH.")
endif()

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------
option(MCPU_USE_POOLED_PROPOSAL
    "Reused proposal State + sparse ProposalPatch" ON)
option(MCPU_EIGEN_HOT_FLAGS
    "Define EIGEN_NO_DEBUG + EIGEN_DONT_PARALLELIZE" OFF)


# ---------------------------------------------------------------------------
# Static library: MCPU_lib  (all C++ physics sources)
# ---------------------------------------------------------------------------
# CONFIGURE_DEPENDS: without it, adding a .cpp under src/ does not re-run
# configure, so the new file is silently NOT compiled until someone wipes the
# build directory by hand.
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")
# CMAKE_CURRENT_SOURCE_DIR, not CMAKE_SOURCE_DIR: the glob above is relative to
# this file, so using the project-root variable here would fail to remove
# bindings.cpp if pyMCPU were ever add_subdirectory()'d, compiling PyInit into
# both targets.
list(REMOVE_ITEM SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/bindings.cpp")

add_library(MCPU_lib STATIC ${SRC_FILES})
target_link_libraries(MCPU_lib PUBLIC Eigen3::Eigen)

# Treat Eigen's headers as system headers so the newly-enabled -Wall -Wextra
# reports OUR code and not a few thousand lines of template diagnostics.
get_target_property(_eigen_inc Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
if(_eigen_inc)
    target_include_directories(MCPU_lib SYSTEM PUBLIC ${_eigen_inc})
endif()
target_include_directories(MCPU_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/pymcpu/reporters/xdrfiles>
)

# ---------------------------------------------------------------------------
# Pybind11 extension: mcpu_core
# ---------------------------------------------------------------------------
pybind11_add_module(mcpu_core src/bindings/bindings.cpp)
target_link_libraries(mcpu_core PRIVATE MCPU_lib)

# RPATH policy: do NOT embed any absolute paths.  Runtime library resolution
# is handled by conda ($CONDA_PREFIX/lib) or module-loaded GCC
# (LD_LIBRARY_PATH).  Hardcoded RPATHs to /n/sw/helmod-rocky8/... made the
# extension non-portable across nodes/environments.
set_target_properties(mcpu_core PROPERTIES
    CXX_STANDARD            20
    CXX_STANDARD_REQUIRED   ON
    CXX_EXTENSIONS          OFF
    SKIP_BUILD_RPATH        TRUE
    BUILD_RPATH             ""
    INSTALL_RPATH           ""
    INSTALL_RPATH_USE_LINK_PATH OFF
)

set_target_properties(MCPU_lib PROPERTIES
    CXX_STANDARD          20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS        OFF
)

# Install rule: scikit-build-core reads this to place the .so into the wheel
# at <platlib>/pymcpu/mcpu_core.<abi-tag>.so.
install(TARGETS mcpu_core LIBRARY DESTINATION pymcpu)

# ---------------------------------------------------------------------------
# Warning flags
# ---------------------------------------------------------------------------
# Advisory, NOT -Werror. Until now these lived only in the Debug branch, so
# every Release build -- i.e. every wheel -- compiled with no warning flags at
# all. Eigen's headers are marked SYSTEM below so its warnings do not drown the
# signal from our own code.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(MCPU_WARN_FLAGS -Wall -Wextra)
else()
    set(MCPU_WARN_FLAGS "")
endif()

# ---------------------------------------------------------------------------
# Build type defaults
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Build type (Release, Debug, RelWithDebInfo)" FORCE)
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" UPPER_BUILD_TYPE)

if(UPPER_BUILD_TYPE STREQUAL "DEBUG")
    message(STATUS "Build Type: DEBUG (-O0 -g)")
    target_compile_options(MCPU_lib  PRIVATE -O0 -g3 -fno-omit-frame-pointer -Wall -Wextra)
    target_compile_options(mcpu_core PRIVATE -O0 -g3 -fno-omit-frame-pointer -Wall -Wextra)
    set(ENABLE_LTO OFF)
    # Debug deliberately carries no -march: it is for stepping through logic,
    # not for reproducing Release numerics. Use -DMCPU_ARCH plus
    # -DCMAKE_BUILD_TYPE=RelWithDebInfo if you need both symbols and the
    # shipping ISA.
    set(MCPU_APPLY_ARCH OFF)
elseif(UPPER_BUILD_TYPE STREQUAL "RELWITHDEBINFO")
    message(STATUS "Build Type: RELWITHDEBINFO (Optimized + Symbols)")
    target_compile_options(MCPU_lib  PRIVATE -O3 -g -fno-omit-frame-pointer ${MCPU_WARN_FLAGS})
    target_compile_options(mcpu_core PRIVATE -O3 -g -fno-omit-frame-pointer ${MCPU_WARN_FLAGS})
    target_compile_definitions(MCPU_lib  PRIVATE NDEBUG)
    target_compile_definitions(mcpu_core PRIVATE NDEBUG)
    set(ENABLE_LTO OFF)
    set(MCPU_APPLY_ARCH ON)
else()
    message(STATUS "Build Type: RELEASE (Max Optimization)")
    target_compile_options(MCPU_lib  PRIVATE -O3 ${MCPU_WARN_FLAGS})
    target_compile_options(mcpu_core PRIVATE -O3 ${MCPU_WARN_FLAGS})
    target_compile_definitions(MCPU_lib  PRIVATE NDEBUG)
    target_compile_definitions(mcpu_core PRIVATE NDEBUG)
    set(MCPU_APPLY_ARCH ON)
    set(ENABLE_LTO ON)
endif()

# ---------------------------------------------------------------------------
# CPU baseline (ISA) selection -- the one knob
# ---------------------------------------------------------------------------
# Spelled as a TIER, not a raw -march value, because the raw spelling is not
# portable across the compilers this project supports: GCC 8.5 -- the oldest
# version verified to build the tree -- REJECTS -march=x86-64-v3 and needs
# -march=haswell for the same ISA level. A tier names the intent and CMake
# finds a spelling this compiler accepts, the same shape as CXX_STANDARD 20
# resolving to -std=c++20 or -std=c++2a.
#
# HISTORY, because this is the bug being fixed: MCPU_ARCH was a plain
# CACHE STRING and CMakeLists.txt read no environment variable at all, while
# README/docs/conda-recipe all documented `MCPU_ARCH=... pip install` as the
# override. A CMake cache variable is NOT populated from the environment, so
# every documented override silently did nothing and built AVX-512 regardless
# -- including the conda recipe whose entire purpose was to avoid AVX-512.
# The $ENV{} initializer below plus cmake.define in pyproject.toml is what
# makes the documented commands real.
if(NOT DEFINED MCPU_ARCH AND DEFINED ENV{MCPU_ARCH} AND NOT "$ENV{MCPU_ARCH}" STREQUAL "")
    set(_mcpu_arch_init "$ENV{MCPU_ARCH}")
else()
    set(_mcpu_arch_init "v3")
endif()
set(MCPU_ARCH "${_mcpu_arch_init}" CACHE STRING
    "CPU baseline: v2 | v3 (default) | v4 | native | none | a raw -march value")
set_property(CACHE MCPU_ARCH PROPERTY STRINGS v2 v3 v4 native none)
if(MCPU_ARCH STREQUAL "")
    set(MCPU_ARCH "v3" CACHE STRING "" FORCE)
endif()

# The cache is read once; the environment is not. A stale build tree silently
# ignoring a changed MCPU_ARCH is the same class of failure as the one above,
# so say it out loud.
if(DEFINED ENV{MCPU_ARCH} AND NOT "$ENV{MCPU_ARCH}" STREQUAL "${MCPU_ARCH}")
    message(WARNING
        "MCPU_ARCH=$ENV{MCPU_ARCH} is set in the environment, but this build "
        "tree is configured with MCPU_ARCH=${MCPU_ARCH} (cached from an earlier "
        "configure). Re-run with -DMCPU_ARCH=$ENV{MCPU_ARCH}, or delete the "
        "build directory.")
endif()

# Tier -> candidate -march spellings, best first. The aliases matter: the exact
# string `x86-64-v3` is already published in README.md, docs/installation.rst,
# docs/known_issues.md, CHANGELOG.md and conda-recipe/meta.yaml, so accepting
# it means every already-documented command starts working rather than starting
# to fail -- and it works on GCC 8.5, which taking it literally could not.
string(TOLOWER "${MCPU_ARCH}" _arch_key)
if(_arch_key MATCHES "^(v2|x86[-_]64[-_]v2)$")
    set(MCPU_ARCH_TIER "v2")
    set(_arch_candidates x86-64-v2 nehalem)
elseif(_arch_key MATCHES "^(v3|x86[-_]64[-_]v3)$")
    set(MCPU_ARCH_TIER "v3")
    set(_arch_candidates x86-64-v3 haswell)
elseif(_arch_key MATCHES "^(v4|x86[-_]64[-_]v4)$")
    set(MCPU_ARCH_TIER "v4")
    set(_arch_candidates x86-64-v4 skylake-avx512)
elseif(_arch_key STREQUAL "native")
    set(MCPU_ARCH_TIER "native")
    set(_arch_candidates native)
elseif(_arch_key MATCHES "^(none|off|generic|baseline)$")
    # For distro/conda packagers who manage -march through CXXFLAGS. Without an
    # opt-out their only recourse is patching this file.
    set(MCPU_ARCH_TIER "none")
    set(_arch_candidates "")
else()
    set(MCPU_ARCH_TIER "custom")
    set(_arch_candidates "${MCPU_ARCH}")
endif()

set(MCPU_ARCH_MARCH "")
set(MCPU_ARCH_FLAGS "")
if(NOT MCPU_APPLY_ARCH)
    message(STATUS "CPU baseline: not applied for this build type (Debug)")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # NOTE: /fp:fast was removed here. It is MSVC's unsafe-math switch, it
    # contradicted the "SAFE math only" status message a few lines below, and it
    # made MSVC builds not FP-comparable with GCC builds. FP semantics are not
    # an architecture decision.
    if(MCPU_ARCH_TIER STREQUAL "v4")
        set(MCPU_ARCH_FLAGS /arch:AVX512)
    elseif(NOT MCPU_ARCH_TIER STREQUAL "none")
        set(MCPU_ARCH_FLAGS /arch:AVX2)
    endif()
    set(MCPU_ARCH_MARCH "${MCPU_ARCH_FLAGS}")
elseif(_arch_candidates)
    include(CheckCXXCompilerFlag)
    foreach(_cand IN LISTS _arch_candidates)
        string(MAKE_C_IDENTIFIER "MCPU_CXX_ACCEPTS_march_${_cand}" _probe)
        check_cxx_compiler_flag("-march=${_cand}" ${_probe})
        if(${_probe})
            set(MCPU_ARCH_MARCH "${_cand}")
            set(MCPU_ARCH_FLAGS "-march=${_cand}")
            break()
        endif()
    endforeach()
    if(NOT MCPU_ARCH_MARCH)
        # Hard error, never a silent downgrade across tiers: a quiet fallback
        # produces a slower binary while the user believes they got what they
        # asked for, which is exactly the failure this block replaces.
        message(FATAL_ERROR
            "MCPU_ARCH=${MCPU_ARCH} (tier '${MCPU_ARCH_TIER}') is not supported by "
            "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}: none of "
            "[${_arch_candidates}] is a valid -march value.\n"
            "  lower tier:   -DMCPU_ARCH=v3    (AVX2 + FMA, GCC >= 4.9)\n"
            "  this machine: -DMCPU_ARCH=native\n"
            "  opt out:      -DMCPU_ARCH=none")
    endif()
    # Compare against the tier's FIRST candidate -- its preferred spelling --
    # not against the tier key. Comparing to "v3" made this fire even when
    # -march=x86-64-v3 was used verbatim, reporting a substitution that had
    # not happened.
    list(GET _arch_candidates 0 _arch_preferred)
    if(NOT MCPU_ARCH_MARCH STREQUAL "${_arch_preferred}"
       AND NOT MCPU_ARCH_TIER STREQUAL "custom"
       AND NOT MCPU_ARCH_TIER STREQUAL "native")
        # Within-tier substitution is fine but must not be silent. Note that
        # `haswell` is a strict SUPERSET of `x86-64-v3` (it adds PCLMUL, RDRND,
        # FSGSBASE, XSAVEOPT) and tunes differently, so the two are equivalent
        # in guaranteed ISA level but not byte-identical in codegen.
        message(STATUS
            "CPU baseline: tier ${MCPU_ARCH_TIER} resolved to -march=${MCPU_ARCH_MARCH} "
            "(this compiler does not accept the x86-64-${MCPU_ARCH_TIER} spelling)")
    endif()
endif()

if(MCPU_ARCH_FLAGS)
    target_compile_options(MCPU_lib  PRIVATE ${MCPU_ARCH_FLAGS})
    target_compile_options(mcpu_core PRIVATE ${MCPU_ARCH_FLAGS})
endif()
message(STATUS "CPU baseline: MCPU_ARCH=${MCPU_ARCH} -> tier ${MCPU_ARCH_TIER} -> ${MCPU_ARCH_FLAGS}")

# ---------------------------------------------------------------------------
# Floating-point contraction
# ---------------------------------------------------------------------------
# Empty (the default) means "do not pass the flag", i.e. inherit the
# compiler's own default -- `fast` for GCC in C++ mode, which fuses
# a*b+c into an FMA with one rounding instead of two. Setting this to `off`
# makes the arithmetic arch-invariant by construction at the cost of FMA
# throughput in the Mu hot loops.
#
# There is no predefined macro reporting -ffp-contract state, so the only way
# build_info() can report it honestly is for us to be the ones who set it.
# Empty is reported as "compiler-default".
set(MCPU_FP_CONTRACT "" CACHE STRING
    "-ffp-contract value: off | on | fast | empty for the compiler default")
# STREQUAL "" rather than a truthiness test: CMake's if(<var>) coerces the
# VALUE to a boolean, and its false constants include the literal strings
# "off", "no", "false" and "0". So `if(MCPU_FP_CONTRACT)` is FALSE when the
# value is exactly "off" -- the one value someone is most likely to pass here.
# That silently skipped the flag while the build appeared to succeed.
if(NOT MCPU_FP_CONTRACT STREQUAL "" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(MCPU_lib  PRIVATE -ffp-contract=${MCPU_FP_CONTRACT})
    target_compile_options(mcpu_core PRIVATE -ffp-contract=${MCPU_FP_CONTRACT})
    set(MCPU_FP_CONTRACT_REPORT "${MCPU_FP_CONTRACT}")
    message(STATUS "FP contraction: -ffp-contract=${MCPU_FP_CONTRACT}")
    # Code with explicit FMA intrinsics (SturmSolver.h) matches the scalar
    # loops only when those contract to FMAs too, so it falls back to them
    # here. GCC treats `on` like `off`, hence "anything but fast".
    if(NOT MCPU_FP_CONTRACT STREQUAL "fast")
        target_compile_definitions(MCPU_lib  PRIVATE MCPU_FP_CONTRACT_OFF=1)
        target_compile_definitions(mcpu_core PRIVATE MCPU_FP_CONTRACT_OFF=1)
    endif()
else()
    set(MCPU_FP_CONTRACT_REPORT "compiler-default")
    message(STATUS "FP contraction: compiler default (GCC C++: fast)")
endif()

# ---------------------------------------------------------------------------
# Eigen compile definitions
# ---------------------------------------------------------------------------
target_compile_definitions(MCPU_lib  PRIVATE EIGEN_DONT_ALIGN_STATICALLY)
target_compile_definitions(mcpu_core PRIVATE EIGEN_DONT_ALIGN_STATICALLY)
if(MCPU_EIGEN_HOT_FLAGS)
    target_compile_definitions(MCPU_lib  PRIVATE EIGEN_NO_DEBUG EIGEN_DONT_PARALLELIZE)
    target_compile_definitions(mcpu_core PRIVATE EIGEN_NO_DEBUG EIGEN_DONT_PARALLELIZE)
endif()

# ---------------------------------------------------------------------------
# Feature-flag compile definitions
# ---------------------------------------------------------------------------
if(MCPU_USE_POOLED_PROPOSAL)
    target_compile_definitions(MCPU_lib  PUBLIC MCPU_USE_POOLED_PROPOSAL=1)
    target_compile_definitions(mcpu_core PUBLIC MCPU_USE_POOLED_PROPOSAL=1)
    message(STATUS "MCPU_USE_POOLED_PROPOSAL=1")
else()
    target_compile_definitions(MCPU_lib  PUBLIC MCPU_USE_POOLED_PROPOSAL=0)
    target_compile_definitions(mcpu_core PUBLIC MCPU_USE_POOLED_PROPOSAL=0)
    message(STATUS "MCPU_USE_POOLED_PROPOSAL=0")
endif()

target_compile_definitions(MCPU_lib  PUBLIC MCPU_UNSAFE_MATH=0)
target_compile_definitions(mcpu_core PUBLIC MCPU_UNSAFE_MATH=0)
target_compile_definitions(MCPU_lib  PUBLIC MCPU_FP_VERIFY_ATOL=0.001)
target_compile_definitions(mcpu_core PUBLIC MCPU_FP_VERIFY_ATOL=0.001)
message(STATUS "SAFE math only (IEEE-ish Release FP)")

# ---------------------------------------------------------------------------
# LTO (Release only)
# ---------------------------------------------------------------------------
set(MCPU_LTO_EFFECTIVE 0)
if(ENABLE_LTO)
    include(CheckIPOSupported)
    # LANGUAGES CXX is load-bearing, not tidiness. Without it
    # check_ipo_supported() checks EVERY enabled language, and FetchContent'd
    # Eigen calls enable_language(Fortran) from its blas/, lapack/ and test/
    # subdirectories. CMake has no IPO support for Fortran, so the check
    # returned false and LTO was silently dropped -- in the default build path
    # only. A direct `cmake` build that finds a system Eigen never trips it,
    # so every local benchmark kept LTO while the wheel, CI and conda-forge
    # builds lost it. Caught by build_info() reporting lto:false, which is why
    # the check below is scoped to LANGUAGES CXX.
    check_ipo_supported(RESULT ipo_supported OUTPUT error LANGUAGES CXX)
    if(ipo_supported)
        set_target_properties(MCPU_lib  PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
        set_target_properties(mcpu_core PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
        # EFFECTIVE, not requested: the request can silently fail the support
        # check below, and the honest report is the conjunction.
        set(MCPU_LTO_EFFECTIVE 1)
        message(STATUS "LTO/IPO enabled")
    else()
        message(STATUS "LTO/IPO not supported: ${error}")
    endif()
endif()

# ---------------------------------------------------------------------------
# Optional C++ tests
# ---------------------------------------------------------------------------
# REMOVED: a BUILD_TESTING block guarding test_scripts/test_cell_list.cpp,
# which has never existed in this repository (`git ls-files` confirms it is
# untracked), so the guard always took its else() branch. It also linked the
# lowercase `eigen` target, which since the find_package(Eigen3) change exists
# only on the FetchContent path -- it would have become a hard configure error
# the first time someone actually added a C++ test.
#
# C++-side verification lives in PhysicsVerifier
# (include/pymcpu/testing/PhysicsVerifier.h), which is bound to Python and
# exercised by tests/physics/test_physics_verifiers.py. There is no separate
# ctest suite.

# ---------------------------------------------------------------------------
# Generated build-configuration header (consumed only by bindings.cpp)
# ---------------------------------------------------------------------------
string(REPLACE ";" " " MCPU_ARCH_FLAGS_STR "${MCPU_ARCH_FLAGS}")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildConfig.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/pymcpu/BuildConfig.h"
    @ONLY)
target_include_directories(mcpu_core PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
