#pragma once
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
// MCPU_FP_CONTRACT_OFF (set by CMake when MCPU_FP_CONTRACT is anything but `fast`):
// there the scalar Horner step rounds twice while _mm256_fmadd_pd rounds once, so the
// packed counts would stop matching the scalar ones that v2/none builds run.
#if defined(__AVX2__) && defined(__FMA__) && !defined(MCPU_FP_CONTRACT_OFF)
#include <immintrin.h>
#define MCPU_STURM_SIMD 1
#else
#define MCPU_STURM_SIMD 0
#endif

namespace TripeptideLoopClosure {

// Sturm-sequence root finder (Graphics Gems 1990, Hook & McAree).
//
// Speed without changing a single root bit. ~88% of the Sturm counts a solve made went
// to the single-root fallback bisection (regula falsi, since replaced, stopped on
// |f(x)/x| < 1e-15, which 40% of roots never met in 20 iterations; each then bisected to
// 1e-15 relative, ~51 counts),
// and each count was 17 dependent scalar Horner loops plus a data-dependent branch per
// sign comparison. With AVX2+FMA (the default v3 build) a count now evaluates four
// sequence members per vector -- the same fused multiply-add per lane that the scalar
// Horner step contracts to under -ffp-contract=fast (GCC's default), zero-padded above
// each member's order, which is exact (x*0 + c == c; counts are only taken at finite
// x) -- and counts the sign changes branch-free with the same `lf == 0 || lf*f < 0`
// test. The fallback bisection takes two steps per round: it evaluates the round's mid
// and both possible next mids, (mid+max)/2 and (min+mid)/2, which are exactly the mids
// the one-step loop would compute, then makes the same two decisions. Roots, their order
// and every decision are unchanged; only the instruction schedule differs.
// MEASURED on the KIC polynomials of the baseline runs (6,378 chignolin and 1,230 actin
// solves): 37.0k -> 14.6k cycles/solve, all 24,182 roots bit-identical. The FMA density
// holds the core at AVX licence 1, so with a full socket (22 processes, Xeon 8268) the
// clock drops 3.44 -> 2.97 GHz: socket throughput -3.4% on actin, +11% on chignolin
// against the scalar counts (+7% / +17% for one process). A 128-bit version kept
// licence 1 (density, not width, sets it) and lost 3.1% there, so the 256-bit one
// stays. Without AVX2 (v2/none builds) the original scalar loops run unchanged.
//
// Single-root refinement (not bit-identical, allowed by the tolerance rule): the
// Sturm counts above still isolate each root, and the root is then finished by Newton's
// method kept inside its sign-change bracket (newton), stopping at the same 1e-15
// relative tolerance. That takes about 6 Horner passes per root where regula falsi plus
// sign halvings took about 50. The Sturm-count bisection runs only when the bracket
// ends have the same sign.
//
// Bookkeeping (roots bit-identical on every input measured): isolation first collects
// every single-root bracket, then newton4 finishes up to four of them at once in AVX
// lanes, each lane making newton()'s decisions with the same operations, so a solve pays
// for its slowest root's iterations rather than the sum. pack() writes the known-order
// layout without per-entry order lookups, and modp() reads each step's leading remainder
// coefficient once so its update loop vectorises. MEASURED on 3,240 KIC polynomials (T4L
// windows, 5,880 roots, all bit-identical): 13.0k -> 9.1k TSC ticks per solve. The lanes
// match the old scalar loop only while the compiler contracts its Horner step to an FMA
// (GCC's default).
class SturmSolver {
private:
    static constexpr int MAX_ORDER = 16;
    static constexpr int MAXPOW = 32;
    static constexpr double SMALL_ENOUGH = 1.0e-18;

    // coef is left uninitialised: every reader stops at ord, and modp copies
    // only coef[0..u.ord]. Zero-filling the 32-member sequence in solve() was a
    // 4.6 KB memset per solve (about 2% of T4L KIC-only cycles).
    struct Poly {
        int ord = 0;
        std::array<double, MAX_ORDER + 1> coef;
    };

    double rel_error = 1.0e-15;
    int max_it = 100;

    // The Sturm sequence packed for vector evaluation: T[g][j] holds coefficient j of
    // members 4g..4g+3, zero above each member's order. Only the full sequence
    // (np == MAX_ORDER, orders 16, 15, ..., 0 -- every solve measured) is packed.
    struct Packed {
#if MCPU_STURM_SIMD
        alignas(32) double T[5][MAX_ORDER + 1][4];
#endif
        bool ok = false;
    };

    int modp(const Poly& u, const Poly& v, Poly& r) const {
        std::copy_n(u.coef.begin(), u.ord + 1, r.coef.begin());
        
        if (v.coef[v.ord] < 0.0) {
            for (int k = u.ord - v.ord - 1; k >= 0; k -= 2) 
                r.coef[k] = -r.coef[k];
            for (int k = u.ord - v.ord; k >= 0; k--) {
                const double lead = r.coef[v.ord + k];
                for (int j = v.ord + k - 1; j >= k; j--)
                    r.coef[j] = -r.coef[j] - lead * v.coef[j - k];
            }
        } else {
            for (int k = u.ord - v.ord; k >= 0; k--) {
                const double lead = r.coef[v.ord + k];
                for (int j = v.ord + k - 1; j >= k; j--)
                    r.coef[j] -= lead * v.coef[j - k];
            }
        }

        int k = v.ord - 1;
        while (k >= 0 && std::abs(r.coef[k]) < SMALL_ENOUGH) {
            r.coef[k] = 0.0;
            k--;
        }
        r.ord = (k < 0) ? 0 : k;
        return r.ord;
    }

    int buildsturm(int ord, std::array<Poly, MAX_ORDER * 2>& sseq) const {
        sseq[0].ord = ord;
        sseq[1].ord = ord - 1;

        double f = std::abs(sseq[0].coef[ord] * ord);
        for (int i = 1; i <= ord; ++i) {
            sseq[1].coef[i - 1] = sseq[0].coef[i] * i / f;
        }

        int np = 1;
        for (int i = 2; i <= ord + 1; ++i) {
            int r_ord = modp(sseq[i - 2], sseq[i - 1], sseq[i]);
            if (r_ord == 0) {
                np = i;
                break;
            }
            f = -std::abs(sseq[i].coef[sseq[i].ord]);
            for (int j = 0; j <= sseq[i].ord; ++j) {
                sseq[i].coef[j] /= f;
            }
            np = i;
        }
        sseq[np].coef[0] = -sseq[np].coef[0];
        return np;
    }

    double evalpoly(int ord, const std::array<double, MAX_ORDER + 1>& coef, double x) const {
        double f = coef[ord];
        for (int i = ord - 1; i >= 0; i--) {
            f = x * f + coef[i];
        }
        return f;
    }

    int numchanges_scalar(int np, const std::array<Poly, MAX_ORDER * 2>& sseq, double a) const {
        int changes = 0;
        double lf = evalpoly(sseq[0].ord, sseq[0].coef, a);
        for (int i = 1; i <= np; i++) {
            double f = evalpoly(sseq[i].ord, sseq[i].coef, a);
            if (lf == 0.0 || lf * f < 0.0) changes++;
            lf = f;
        }
        return changes;
    }

#if MCPU_STURM_SIMD
    // np == MAX_ORDER means every remainder dropped exactly one degree, so member k has
    // order MAX_ORDER - k; group g is read from row MAX_ORDER - 4g down. Writing only
    // those rows with compile-time zero padding replaced a 340-entry loop that looked up
    // each member's order (about a fifth of a solve with roots). Same values, same layout.
    static void pack(int np, const std::array<Poly, MAX_ORDER * 2>& sseq, Packed& P) {
        P.ok = (np == MAX_ORDER);
        if (!P.ok) return;
#pragma GCC unroll 4
        for (int g = 0; g < 4; ++g) {
#pragma GCC unroll 17
            for (int j = 0; j <= MAX_ORDER - 4 * g; ++j)
#pragma GCC unroll 4
                for (int l = 0; l < 4; ++l) {
                    const int k = 4 * g + l;
                    P.T[g][j][l] = (j <= MAX_ORDER - k)
                                       ? sseq[static_cast<size_t>(k)].coef[static_cast<size_t>(j)] : 0.0;
                }
        }
        P.T[4][0][0] = sseq[MAX_ORDER].coef[0];
        P.T[4][0][1] = P.T[4][0][2] = P.T[4][0][3] = 0.0;
    }

    // Sign changes between consecutive members f_0..f_16 held in F[0..16].
    static int count_changes(const double* F) {
        const __m256d z = _mm256_setzero_pd();
        int changes = 0;
        for (int i = 0; i < MAX_ORDER; i += 4) {
            const __m256d prev = _mm256_loadu_pd(F + i);
            const __m256d cur = _mm256_loadu_pd(F + i + 1);
            const __m256d eq0 = _mm256_cmp_pd(prev, z, _CMP_EQ_OQ);
            const __m256d neg = _mm256_cmp_pd(_mm256_mul_pd(prev, cur), z, _CMP_LT_OQ);
            changes += __builtin_popcount(
                static_cast<unsigned>(_mm256_movemask_pd(_mm256_or_pd(eq0, neg))));
        }
        return changes;
    }

    // Sturm counts at NPT points at once. Group g (degree 16 - 4g) joins the Horner
    // recurrence when j reaches its degree, so every lane sees exactly its scalar loop.
    template <int NPT>
    static void numchanges_simd(const Packed& P, const double* xs, int* out) {
        __m256d x[NPT], f[4][NPT];
#pragma GCC unroll 4
        for (int p = 0; p < NPT; ++p) x[p] = _mm256_set1_pd(xs[p]);
#pragma GCC unroll 4
        for (int g = 0; g < 4; ++g) {
            const __m256d top = _mm256_load_pd(P.T[g][MAX_ORDER - 4 * g]);
#pragma GCC unroll 4
            for (int p = 0; p < NPT; ++p) f[g][p] = top;
        }
#pragma GCC unroll 16
        for (int j = MAX_ORDER - 1; j >= 0; --j) {
#pragma GCC unroll 4
            for (int g = 0; g < 4; ++g) {
                if (j >= MAX_ORDER - 4 * g) continue;   // group g not started yet
                const __m256d t = _mm256_load_pd(P.T[g][j]);
#pragma GCC unroll 4
                for (int p = 0; p < NPT; ++p) f[g][p] = _mm256_fmadd_pd(x[p], f[g][p], t);
            }
        }
        const __m256d last = _mm256_load_pd(P.T[4][0]);
#pragma GCC unroll 4
        for (int p = 0; p < NPT; ++p) {
            alignas(32) double F[20];
            for (int g = 0; g < 4; ++g) _mm256_store_pd(F + 4 * g, f[g][p]);
            _mm256_store_pd(F + 16, last);
            out[p] = count_changes(F);
        }
    }
#endif

    int numchanges(int np, const std::array<Poly, MAX_ORDER * 2>& sseq, const Packed& P, double a) const {
#if MCPU_STURM_SIMD
        if (P.ok) {
            int c;
            numchanges_simd<1>(P, &a, &c);
            return c;
        }
#else
        (void)P;
#endif
        return numchanges_scalar(np, sseq, a);
    }

    int numroots(int np, const std::array<Poly, MAX_ORDER * 2>& sseq, int& atneg, int& atpos) const {
        int atposinf = 0, atneginf = 0;
        double lf = sseq[0].coef[sseq[0].ord];
        
        for (int i = 1; i <= np; i++) {
            double f = sseq[i].coef[sseq[i].ord];
            if (lf == 0.0 || lf * f < 0.0) atposinf++;
            lf = f;
        }

        lf = (sseq[0].ord & 1) ? -sseq[0].coef[sseq[0].ord] : sseq[0].coef[sseq[0].ord];
        
        for (int i = 1; i <= np; i++) {
            double f = (sseq[i].ord & 1) ? -sseq[i].coef[sseq[i].ord] : sseq[i].coef[sseq[i].ord];
            if (lf == 0.0 || lf * f < 0.0) atneginf++;
            lf = f;
        }

        atneg = atneginf;
        atpos = atposinf;
        return atneginf - atposinf;
    }

    // Finishes the single root in [a, b] by Newton's method on the polynomial, kept inside
    // the sign-change bracket (Numerical Recipes rtsafe): a step that would leave the
    // bracket, or that does not halve the previous step, is a halving instead. Returns
    // false, touching nothing, when f(a) and f(b) have the same sign. Stops when a step is
    // below the 1e-15 relative tolerance, f is exactly 0, or the bracket meets the
    // bisection stop width. Each iteration is one Horner pass for f and f' together; it
    // replaced Illinois regula falsi with |f(x)/x| < 1e-15, which 40% of roots never met
    // in 20 iterations, followed by ~30 sign halvings.
    bool newton(int ord, const std::array<double, MAX_ORDER + 1>& coef, double a, double b, double& root) const {
        double fa = coef[ord], fb = coef[ord];
        for (int i = ord - 1; i >= 0; i--) {
            fa = a * fa + coef[i];
            fb = b * fb + coef[i];
        }
        if (fa * fb > 0.0) return false;
        if (fa == 0.0) { root = a; return true; }
        if (fb == 0.0) { root = b; return true; }
        double lo = a, hi = b;              // f(lo) < 0 < f(hi)
        if (fa > 0.0) std::swap(lo, hi);
        double x = (fb * a - fa * b) / (fb - fa);    // secant start
        if (!(x > std::min(a, b) && x < std::max(a, b))) x = 0.5 * (a + b);
        double dxold = std::abs(b - a), dx = dxold;
        bool newton_last = false;
        for (int its = 0; its < max_it; its++) {
            double f = coef[ord], df = 0.0;
            for (int i = ord - 1; i >= 0; i--) {
                df = x * df + f;
                f = x * f + coef[i];
            }
            if (f == 0.0) { root = x; return true; }
            if (f < 0.0) lo = x; else hi = x;
            const double step = f / df;
            const double xn = x - step;
            // Converged. Tested before the bracket test: a step below one ulp leaves xn == x,
            // which is a bracket end, so the bracket test would otherwise halve the bracket.
            if (bisect_done(0.0, step, x)) { root = xn; return true; }
            if (!(std::abs(2.0 * f) < std::abs(dxold * df)) ||
                !(xn > std::min(lo, hi) && xn < std::max(lo, hi))) {
                // Rejected right after a Newton step below 1e-11 relative: x is at the
                // rounding-noise floor of f. Halving the bracket instead, whose far end may
                // not have moved since the start, would take dozens of passes.
                if (newton_last && bisect_done(0.0, 1.0e-4 * dx, x)) { root = x; return true; }
                newton_last = false;
                dxold = dx;
                dx = 0.5 * (hi - lo);
                x = lo + dx;
            } else {
                newton_last = true;
                dxold = dx;
                dx = step;
                x = xn;
            }
            if (bisect_done(0.0, dx, x) || bisect_done(lo, hi, 0.5 * (lo + hi))) { root = x; return true; }
        }
        root = x;
        return true;
    }

    // The single-root loop's stop test; it does not depend on the count at mid.
    bool bisect_done(double min, double max, double mid) const {
        if (std::abs(mid) > rel_error) return std::abs((max - min) / mid) < rel_error;
        return std::abs(max - min) < rel_error;
    }

    // Isolated roots, in ascending order. A leaf is either a bracket holding exactly one
    // root (have == false until it is finished) or a value the isolation already settled
    // (have == true: a multi-root interval that hit max_it). A polynomial of degree 16
    // has at most 16 roots, but with tightly clustered roots the floating-point Sturm
    // counts stop being monotone and isolation can emit hundreds of leaves (837 seen on a
    // synthetic input). The first 17 live in the inline array; past that the leaves move
    // to a heap vector, so every leaf is kept and the output matches the old solver.
    struct Leaf {
        double min, max, val;
        int atmin;
        bool have;
    };
    struct Leaves {
        std::array<Leaf, MAX_ORDER + 1> inl;
        std::vector<Leaf> spill;
        Leaf* v = inl.data();
        int n = 0;
        Leaves() = default;
        Leaves(const Leaves&) = delete;
        Leaves& operator=(const Leaves&) = delete;
        void add(double mn, double mx, int at, double val, bool have) {
            if (n < static_cast<int>(inl.size())) {
                inl[static_cast<size_t>(n++)] = Leaf{mn, mx, val, at, have};
                return;
            }
            // Not taken on any real KIC input measured.
            if (spill.empty()) spill.assign(inl.begin(), inl.end());
            spill.push_back(Leaf{mn, mx, val, at, have});
            v = spill.data();
            ++n;
        }
    };

    // Isolation only: records each single-root bracket instead of finishing it, so that
    // finish() can run Newton on up to four brackets at once.
    void sbisect(int np, const std::array<Poly, MAX_ORDER * 2>& sseq, const Packed& P, double min, double max, int atmin, int atmax, Leaves& out) const {
        double mid = 0.0;
        if (atmin - atmax == 1) { out.add(min, max, atmin, 0.0, false); return; }

        for (int its = 0; its < max_it; its++) {
            mid = (min + max) / 2.0;
            int atmid = numchanges(np, sseq, P, mid);
            int n1 = atmin - atmid;
            int n2 = atmid - atmax;

            if (n1 != 0 && n2 != 0) {
                sbisect(np, sseq, P, min, mid, atmin, atmid, out);
                sbisect(np, sseq, P, mid, max, atmid, atmax, out);
                return;
            }
            if (n1 == 0) min = mid;
            else max = mid;
        }

        for (int n1 = atmax; n1 < atmin; n1++) out.add(min, max, atmin, mid, true);
    }

    // Sturm-count bisection of a single-root bracket whose ends have the same sign
    // (Newton cannot start there).
    double bisect_one(int np, const std::array<Poly, MAX_ORDER * 2>& sseq, const Packed& P, double min, double max, int atmin) const {
        double mid = 0.0;
        int its = 0;
#if MCPU_STURM_SIMD
        // Two of the loop below per round; see the class comment.
        while (P.ok && its + 2 <= max_it) {
            mid = (min + max) / 2.0;
            if (bisect_done(min, max, mid)) return mid;
            const double xs[3] = {mid, (mid + max) / 2.0, (min + mid) / 2.0};
            int c[3];
            numchanges_simd<3>(P, xs, c);
            double mid2;
            int at2;
            if ((atmin - c[0]) == 0) { min = mid; mid2 = xs[1]; at2 = c[1]; }
            else                     { max = mid; mid2 = xs[2]; at2 = c[2]; }
            mid = mid2;
            if (bisect_done(min, max, mid)) return mid;
            if ((atmin - at2) == 0) min = mid;
            else max = mid;
            its += 2;
        }
#endif
        for (; its < max_it; its++) {
            mid = (min + max) / 2.0;
            int atmid = numchanges(np, sseq, P, mid);
            
            if (std::abs(mid) > rel_error) {
                if (std::abs((max - min) / mid) < rel_error) return mid;
            } else if (std::abs(max - min) < rel_error) {
                return mid;
            }

            if ((atmin - atmid) == 0) min = mid;
            else max = mid;
        }
        return mid;
    }

#if MCPU_STURM_SIMD
    // newton() on up to four brackets at once, one per AVX lane. Each lane makes the
    // decisions newton() makes, with the same operations: the Horner steps are the fused
    // multiply-adds the scalar loop contracts to, the bracket set-up is newton()'s own
    // scalar code, and a finished lane is frozen by blends. A solve with several roots
    // then pays for the slowest root's iterations once instead of every root's in turn;
    // each iteration is one dependent Horner chain (~16 FMA latencies) either way.
    void newton4(int ord, const std::array<double, MAX_ORDER + 1>& coef, Leaves& L, const int* idx, int m) const {
        alignas(32) double A[4], B[4], FA[4], FB[4], X[4], LO[4], HI[4], DX[4], R[4];
        for (int l = 0; l < 4; ++l) {
            const Leaf& lf = L.v[static_cast<size_t>(idx[l < m ? l : 0])];
            A[l] = lf.min;
            B[l] = lf.max;
        }
        {
            const __m256d a = _mm256_load_pd(A), b = _mm256_load_pd(B);
            __m256d fa = _mm256_set1_pd(coef[static_cast<size_t>(ord)]), fb = fa;
            for (int i = ord - 1; i >= 0; i--) {
                const __m256d c = _mm256_set1_pd(coef[static_cast<size_t>(i)]);
                fa = _mm256_fmadd_pd(a, fa, c);
                fb = _mm256_fmadd_pd(b, fb, c);
            }
            _mm256_store_pd(FA, fa);
            _mm256_store_pd(FB, fb);
        }
        alignas(32) long long act[4] = {0, 0, 0, 0};
        for (int l = 0; l < 4; ++l) { X[l] = 0.5; LO[l] = 0.0; HI[l] = 1.0; DX[l] = 1.0; }
        for (int l = 0; l < m; ++l) {
            Leaf& lf = L.v[static_cast<size_t>(idx[l])];
            const double a = A[l], b = B[l], fa = FA[l], fb = FB[l];
            if (fa * fb > 0.0) continue;                 // left for bisect_one
            if (fa == 0.0) { lf.val = a; lf.have = true; continue; }
            if (fb == 0.0) { lf.val = b; lf.have = true; continue; }
            double lo = a, hi = b;
            if (fa > 0.0) std::swap(lo, hi);
            double x = (fb * a - fa * b) / (fb - fa);
            if (!(x > std::min(a, b) && x < std::max(a, b))) x = 0.5 * (a + b);
            X[l] = x; LO[l] = lo; HI[l] = hi; DX[l] = std::abs(b - a);
            act[l] = -1;
        }
        __m256d active = _mm256_castsi256_pd(_mm256_load_si256(reinterpret_cast<const __m256i*>(act)));
        if (_mm256_movemask_pd(active) == 0) return;

        const __m256d zero = _mm256_setzero_pd(), sign = _mm256_set1_pd(-0.0);
        const __m256d half = _mm256_set1_pd(0.5), two = _mm256_set1_pd(2.0);
        const __m256d tiny = _mm256_set1_pd(1.0e-4), rel = _mm256_set1_pd(rel_error);
        const __m256d ones = _mm256_castsi256_pd(_mm256_set1_epi64x(-1));
        auto vabs = [&](__m256d v) { return _mm256_andnot_pd(sign, v); };
        auto done = [&](__m256d mn, __m256d mx, __m256d mid) {   // bisect_done per lane
            const __m256d d = _mm256_sub_pd(mx, mn);
            const __m256d big = _mm256_cmp_pd(vabs(mid), rel, _CMP_GT_OQ);
            const __m256d r1 = _mm256_cmp_pd(vabs(_mm256_div_pd(d, mid)), rel, _CMP_LT_OQ);
            const __m256d r2 = _mm256_cmp_pd(vabs(d), rel, _CMP_LT_OQ);
            return _mm256_blendv_pd(r2, r1, big);
        };
        __m256d x = _mm256_load_pd(X), lo = _mm256_load_pd(LO), hi = _mm256_load_pd(HI);
        __m256d dx = _mm256_load_pd(DX), dxold = dx, nl = zero, root = x;
        const __m256d top = _mm256_set1_pd(coef[static_cast<size_t>(ord)]);
        for (int its = 0; its < max_it && _mm256_movemask_pd(active) != 0; its++) {
            __m256d f = top, df = zero;
            for (int i = ord - 1; i >= 0; i--) {
                df = _mm256_fmadd_pd(x, df, f);
                f = _mm256_fmadd_pd(x, f, _mm256_set1_pd(coef[static_cast<size_t>(i)]));
            }
            const __m256d f0 = _mm256_and_pd(active, _mm256_cmp_pd(f, zero, _CMP_EQ_OQ));
            root = _mm256_blendv_pd(root, x, f0);
            active = _mm256_andnot_pd(f0, active);
            const __m256d neg = _mm256_cmp_pd(f, zero, _CMP_LT_OQ);
            lo = _mm256_blendv_pd(lo, x, _mm256_and_pd(active, neg));
            hi = _mm256_blendv_pd(hi, x, _mm256_andnot_pd(neg, active));
            const __m256d step = _mm256_div_pd(f, df);
            const __m256d xn = _mm256_sub_pd(x, step);
            const __m256d d1 = _mm256_and_pd(active, done(zero, step, x));
            root = _mm256_blendv_pd(root, xn, d1);
            active = _mm256_andnot_pd(d1, active);
            const __m256d mn = _mm256_blendv_pd(lo, hi, _mm256_cmp_pd(hi, lo, _CMP_LT_OQ));   // std::min
            const __m256d mx = _mm256_blendv_pd(lo, hi, _mm256_cmp_pd(lo, hi, _CMP_LT_OQ));   // std::max
            const __m256d ok = _mm256_and_pd(
                _mm256_cmp_pd(vabs(_mm256_mul_pd(two, f)), vabs(_mm256_mul_pd(dxold, df)), _CMP_LT_OQ),
                _mm256_and_pd(_mm256_cmp_pd(xn, mn, _CMP_GT_OQ), _mm256_cmp_pd(xn, mx, _CMP_LT_OQ)));
            __m256d bis = _mm256_andnot_pd(ok, active);
            const __m256d d2 = _mm256_and_pd(bis, _mm256_and_pd(nl, done(zero, _mm256_mul_pd(tiny, dx), x)));
            root = _mm256_blendv_pd(root, x, d2);
            active = _mm256_andnot_pd(d2, active);
            bis = _mm256_andnot_pd(d2, bis);
            const __m256d newt = _mm256_and_pd(ok, active);
            const __m256d hdx = _mm256_mul_pd(half, _mm256_sub_pd(hi, lo));
            dxold = _mm256_blendv_pd(dxold, dx, active);
            dx = _mm256_blendv_pd(_mm256_blendv_pd(dx, hdx, bis), step, newt);
            x = _mm256_blendv_pd(_mm256_blendv_pd(x, _mm256_add_pd(lo, hdx), bis), xn, newt);
            nl = _mm256_blendv_pd(_mm256_blendv_pd(nl, zero, bis), ones, newt);
            const __m256d d3 = _mm256_and_pd(active, _mm256_or_pd(done(zero, dx, x),
                                                     done(lo, hi, _mm256_mul_pd(half, _mm256_add_pd(lo, hi)))));
            root = _mm256_blendv_pd(root, x, d3);
            active = _mm256_andnot_pd(d3, active);
        }
        root = _mm256_blendv_pd(root, x, active);         // max_it reached: newton() keeps x
        _mm256_store_pd(R, root);
        for (int l = 0; l < m; ++l)
            if (act[l]) {
                Leaf& lf = L.v[static_cast<size_t>(idx[l])];
                lf.val = R[l];
                lf.have = true;
            }
    }
#endif

    // Finishes every single-root bracket by Newton (four at a time with AVX2), falls back
    // to Sturm bisection where the bracket ends have the same sign, and emits the roots in
    // isolation order.
    void finish(int np, const std::array<Poly, MAX_ORDER * 2>& sseq, const Packed& P, Leaves& L, std::vector<double>& roots) const {
        const int ord = sseq[0].ord;
#if MCPU_STURM_SIMD
        int idx[4];
        int m = 0;
        for (int i = 0; i < L.n; ++i) {
            if (L.v[static_cast<size_t>(i)].have) continue;
            idx[m++] = i;
            if (m == 4) { newton4(ord, sseq[0].coef, L, idx, m); m = 0; }
        }
        if (m) newton4(ord, sseq[0].coef, L, idx, m);
#else
        for (int i = 0; i < L.n; ++i) {
            Leaf& lf = L.v[static_cast<size_t>(i)];
            if (!lf.have) lf.have = newton(ord, sseq[0].coef, lf.min, lf.max, lf.val);
        }
#endif
        for (int i = 0; i < L.n; ++i) {
            Leaf& lf = L.v[static_cast<size_t>(i)];
            if (!lf.have) lf.val = bisect_one(np, sseq, P, lf.min, lf.max, lf.atmin);
            roots.push_back(lf.val);
        }
    }

public:
    void solve(const Eigen::Matrix<double, 17, 1>& poly_coeffs, std::vector<double>& roots) const {
        roots.clear();
        std::array<Poly, MAX_ORDER * 2> sseq;
        sseq[0].ord = MAX_ORDER;
        
        for (int i = MAX_ORDER; i >= 0; i--) {
            sseq[0].coef[i] = poly_coeffs[i];
        }

        int np = buildsturm(MAX_ORDER, sseq);
        int atmin, atmax;
        int nroots = numroots(np, sseq, atmin, atmax);

        if (nroots == 0) return;

        Packed P;
#if MCPU_STURM_SIMD
        pack(np, sseq, P);
#endif

        double min = -1.0;
        int nchanges = numchanges(np, sseq, P, min);
        for (int i = 0; nchanges != atmin && i != MAXPOW; i++) {
            min *= 10.0;
            nchanges = numchanges(np, sseq, P, min);
        }
        atmin = nchanges;

        double max = 1.0;
        nchanges = numchanges(np, sseq, P, max);
        for (int i = 0; nchanges != atmax && i != MAXPOW; i++) {
            max *= 10.0;
            nchanges = numchanges(np, sseq, P, max);
        }
        atmax = nchanges;

        Leaves leaves;
        sbisect(np, sseq, P, min, max, atmin, atmax, leaves);
        finish(np, sseq, P, leaves, roots);
    }
};

} // namespace TripeptideLoopClosure

#undef MCPU_STURM_SIMD
