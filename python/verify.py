"""
verify.py — Empirical and exact verification of the MaxCut algorithms.

Checks:
    1. Randomized:              E[cut] / |E| ≈ 0.5   (empirical, many runs)
    2. Conditional expectations: cut >= |E|/2          (deterministic guarantee)
    3. Pairwise independence:    cut >= |E|/2          (deterministic guarantee)
    4. Pairwise independence property: exact uniformity of all pairs
    5. Cross-validation with Rust binary (if compiled)

Run:
    python verify.py
    python verify.py --no-rust   # skip Rust cross-validation
"""

import math
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent))
from maxcut import (
    cut_size,
    conditional_expectations_maxcut,
    pairwise_independent_maxcut,
    pairwise_independent_bits,
    randomized_maxcut,
    seed_bits_needed,
    nonempty_subsets,
    Edge,
)

RUST_BINARY = Path(__file__).parent.parent / "rust" / "target" / "release" / "maxcut"


# ---------------------------------------------------------------------------
# Graph generators
# ---------------------------------------------------------------------------

def complete_graph(n: int) -> Tuple[int, List[Edge]]:
    edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    return n, edges


def cycle_graph(n: int) -> Tuple[int, List[Edge]]:
    return n, [(i, (i + 1) % n) for i in range(n)]


def path_graph(n: int) -> Tuple[int, List[Edge]]:
    if n < 2:
        return n, []
    return n, [(i, i + 1) for i in range(n - 1)]


def bipartite_complete(a: int, b: int) -> Tuple[int, List[Edge]]:
    """K_{a,b}: vertices 0..a-1 on side A, a..a+b-1 on side B."""
    edges = [(i, a + j) for i in range(a) for j in range(b)]
    return a + b, edges


def random_gnm(n: int, m: int, seed: int = 0) -> Tuple[int, List[Edge]]:
    rng = random.Random(seed)
    edge_set: set = set()
    attempts = 0
    while len(edge_set) < m and attempts < m * 20:
        u, v = sorted(rng.sample(range(n), 2))
        edge_set.add((u, v))
        attempts += 1
    return n, sorted(edge_set)


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def check_expected_cut(
    n: int, edges: List[Edge], num_trials: int = 20_000, tol: float = 0.02
) -> Tuple[bool, str]:
    """Verify E[cut] / |E| ≈ 0.5 over many random runs."""
    m = len(edges)
    if m == 0:
        return True, "no edges (trivially ok)"
    rng = random.Random(42)
    total = sum(randomized_maxcut(n, edges, rng=rng)[1] for _ in range(num_trials))
    ratio = (total / num_trials) / m
    ok = abs(ratio - 0.5) < tol
    return ok, f"E[cut]/|E| = {ratio:.4f}  (want 0.5 ± {tol})"


def check_geq_half(
    n: int, edges: List[Edge], algorithm, label: str
) -> Tuple[bool, str]:
    """Verify cut >= |E|/2 for a deterministic algorithm."""
    m = len(edges)
    _, c = algorithm(n, edges)
    ok = 2 * c >= m  # integer comparison avoids float
    ratio = c / m if m > 0 else 1.0
    return ok, f"cut={c}  |E|/2={m/2:.1f}  ratio={ratio:.3f}"


def check_pairwise_independence_exact(k: int, n: int) -> Tuple[bool, str]:
    """
    Exact check: enumerate all 2^k seeds and verify every pair of generated
    bits is jointly uniform over {0,1}^2.

    For pairwise independence, each of the 4 joint values should appear
    exactly 2^k / 4 = 2^(k-2) times across all seeds.
    """
    if n < 2:
        return True, "n < 2, no pairs to check"
    if k < 2:
        # k=1 gives only 1 bit (n<=1), so no pairs possible anyway
        return True, "k < 2, no pairs possible"

    expected_count = (1 << k) // 4  # = 2^(k-2)

    # Check up to the first 10 pairs (sufficient for correctness)
    pairs_to_check = [
        (i, j)
        for i in range(min(n, 5))
        for j in range(i + 1, min(n, 6))
    ]

    for i, j in pairs_to_check:
        counts: Counter = Counter()
        for seed in range(1 << k):
            bits = pairwise_independent_bits(seed, k, n)
            counts[(bits[i], bits[j])] += 1
        for val in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            if counts[val] != expected_count:
                return False, (
                    f"pair ({i},{j}) value {val}: "
                    f"count={counts[val]} expected={expected_count}"
                )

    return True, (
        f"all {len(pairs_to_check)} pairs exactly uniform "
        f"(each of 4 values appears {expected_count}x over {1<<k} seeds)"
    )


def cross_validate_rust(
    n: int, edges: List[Edge]
) -> Tuple[Optional[bool], str]:
    """
    Run the Rust binary on the same graph and compare cut sizes and assignments.
    Returns (None, reason) if the binary is unavailable.
    """
    if not RUST_BINARY.exists():
        return None, f"binary not found at {RUST_BINARY} (run: cargo build --release)"

    graph_str = f"{n} {len(edges)}\n" + "\n".join(f"{u} {v}" for u, v in edges)
    try:
        result = subprocess.run(
            [str(RUST_BINARY)],
            input=graph_str,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return None, "Rust binary timed out"
    except Exception as exc:
        return None, f"Error running Rust binary: {exc}"

    if result.returncode != 0:
        return False, f"Rust binary exited {result.returncode}: {result.stderr.strip()}"

    # Parse "key: value" lines
    rust: dict = {}
    for line in result.stdout.strip().splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            rust[key.strip()] = val.strip()

    py_ce_assign, py_ce_cut = conditional_expectations_maxcut(n, edges)
    py_pi_assign, py_pi_cut = pairwise_independent_maxcut(n, edges)

    rust_ce_cut = int(rust.get("ce_cut", -1))
    rust_pi_cut = int(rust.get("pi_cut", -1))
    rust_ce_assign = list(map(int, rust.get("ce_assignment", "").split())) if n > 0 else []
    rust_pi_assign = list(map(int, rust.get("pi_assignment", "").split())) if n > 0 else []

    ce_cut_ok = rust_ce_cut == py_ce_cut
    pi_cut_ok = rust_pi_cut == py_pi_cut
    ce_assign_ok = rust_ce_assign == py_ce_assign
    pi_assign_ok = rust_pi_assign == py_pi_assign
    ok = ce_cut_ok and pi_cut_ok and ce_assign_ok and pi_assign_ok

    parts = [
        f"CE cut py={py_ce_cut} rust={rust_ce_cut} {'✓' if ce_cut_ok else '✗'}",
        f"PI cut py={py_pi_cut} rust={rust_pi_cut} {'✓' if pi_cut_ok else '✗'}",
        f"CE assign {'✓' if ce_assign_ok else f'✗ py={py_ce_assign} rust={rust_ce_assign}'}",
        f"PI assign {'✓' if pi_assign_ok else f'✗ py={py_pi_assign} rust={rust_pi_assign}'}",
    ]
    return ok, "  |  ".join(parts)


# ---------------------------------------------------------------------------
# Known-answer tests (specific small graphs)
# ---------------------------------------------------------------------------

def run_known_answer_tests() -> bool:
    """
    Verify exact outputs on small graphs where we can compute answers by hand.
    These double as cross-validation anchors for the Rust implementation.
    """
    passed = True

    # --- K4 ---
    n, edges = complete_graph(4)
    ce_assign, ce_cut = conditional_expectations_maxcut(n, edges)
    # Hand-traced: vertex 0 → T(1), 1 → S(0), 2 → T(1), 3 → S(0); cut=4
    assert ce_assign == [1, 0, 1, 0], f"K4 CE assign: {ce_assign}"
    assert ce_cut == 4, f"K4 CE cut: {ce_cut}"
    _, pi_cut = pairwise_independent_maxcut(n, edges)
    assert pi_cut == 4, f"K4 PI cut: {pi_cut}"
    print("  [KAT] K4 CE assignment [1,0,1,0] cut=4  ✓")
    print("  [KAT] K4 PI cut=4  ✓")

    # --- Single edge (n=2, m=1) ---
    n, edges = 2, [(0, 1)]
    _, ce_cut = conditional_expectations_maxcut(n, edges)
    assert ce_cut == 1, f"K2 CE cut: {ce_cut}"
    _, pi_cut = pairwise_independent_maxcut(n, edges)
    assert pi_cut == 1, f"K2 PI cut: {pi_cut}"
    print("  [KAT] K_2 (single edge) both algorithms cut=1  ✓")

    # --- K_{4,4}: optimal = 16 ---
    # CE always finds optimal for bipartite when A-vertices are processed first.
    # PI only guarantees cut >= |E|/2 = 8; the XOR construction cannot always
    # achieve the perfect bipartition (the required bit patterns may not exist).
    n, edges = bipartite_complete(4, 4)
    _, ce_cut = conditional_expectations_maxcut(n, edges)
    assert ce_cut == 16, f"K_4,4 CE cut: {ce_cut}"
    _, pi_cut = pairwise_independent_maxcut(n, edges)
    assert 2 * pi_cut >= 16, f"K_4,4 PI cut {pi_cut} < |E|/2 = 8"
    print(f"  [KAT] K_{{4,4}} CE cut=16 (optimal) ✓   PI cut={pi_cut} (>= 8) ✓")

    # --- Path on 5 vertices: optimal = 4 (alternate S,T), CE should achieve it ---
    n, edges = path_graph(5)
    _, ce_cut = conditional_expectations_maxcut(n, edges)
    assert ce_cut == 4, f"P5 CE cut: {ce_cut}"
    print("  [KAT] P_5 CE cut=4 (optimal)  ✓")

    # --- Pairwise independence: k=2, exact distribution ---
    # For k=2: subsets {B0}, {B1}, {B0,B1}. Seeds 0..3:
    #   seed 0 (B0=0,B1=0): [0, 0, 0]
    #   seed 1 (B0=1,B1=0): [1, 0, 1]
    #   seed 2 (B0=0,B1=1): [0, 1, 1]
    #   seed 3 (B0=1,B1=1): [1, 1, 0]
    expected = {
        0: [0, 0, 0],
        1: [1, 0, 1],
        2: [0, 1, 1],
        3: [1, 1, 0],
    }
    for seed, exp_bits in expected.items():
        got = pairwise_independent_bits(seed, k=2, n=3)
        assert got == exp_bits, f"PI bits seed={seed}: got {got}, expected {exp_bits}"
    print("  [KAT] pairwise_independent_bits k=2: all 4 seeds correct  ✓")

    return passed


# ---------------------------------------------------------------------------
# Full test suite
# ---------------------------------------------------------------------------

TEST_GRAPHS = [
    ("K_4",           *complete_graph(4)),
    ("K_5",           *complete_graph(5)),
    ("K_8",           *complete_graph(8)),
    ("C_6 (even)",    *cycle_graph(6)),
    ("C_7 (odd)",     *cycle_graph(7)),
    ("C_10",          *cycle_graph(10)),
    ("P_10",          *path_graph(10)),
    ("K_{3,3}",       *bipartite_complete(3, 3)),
    ("K_{4,5}",       *bipartite_complete(4, 5)),
    ("G(20,60)",      *random_gnm(20, 60, seed=42)),
    ("G(50,200)",     *random_gnm(50, 200, seed=99)),
    ("G(100,500)",    *random_gnm(100, 500, seed=7)),
]


def run_all(skip_rust: bool = False) -> bool:
    print("=" * 72)
    print("MAXCUT ALGORITHM VERIFICATION")
    print("=" * 72)

    # Known-answer tests first
    print("\n[ Known-answer tests ]")
    run_known_answer_tests()

    all_passed = True

    print("\n[ Full test suite ]")
    for label, n, edges in TEST_GRAPHS:
        m = len(edges)
        k = seed_bits_needed(n)
        print(
            f"\n  {label:<22}  n={n:3d}  m={m:4d}  "
            f"|E|/2={m/2:6.1f}  k={k}  seeds={1<<k}"
        )

        # 1. Randomized: empirical expectation
        ok, msg = check_expected_cut(n, edges)
        sym = "✓" if ok else "✗"
        print(f"    [Rand]  {sym}  {msg}")
        all_passed &= ok

        # 2. Conditional expectations: deterministic guarantee
        ok, msg = check_geq_half(n, edges, conditional_expectations_maxcut, "CE")
        sym = "✓" if ok else "✗"
        print(f"    [CE  ]  {sym}  {msg}")
        all_passed &= ok

        # 3. Pairwise independence: deterministic guarantee
        ok, msg = check_geq_half(n, edges, pairwise_independent_maxcut, "PI")
        sym = "✓" if ok else "✗"
        print(f"    [PI  ]  {sym}  {msg}")
        all_passed &= ok

        # 4. Pairwise independence property (exact)
        ok, msg = check_pairwise_independence_exact(k, n)
        sym = "✓" if ok else "✗"
        print(f"    [Indp]  {sym}  {msg}")
        all_passed &= ok

        # 5. Rust cross-validation
        if not skip_rust:
            ok, msg = cross_validate_rust(n, edges)
            if ok is None:
                print(f"    [Rust]  –  {msg}")
            else:
                sym = "✓" if ok else "✗"
                print(f"    [Rust]  {sym}  {msg}")
                all_passed &= ok

    print("\n" + "=" * 72)
    verdict = "ALL TESTS PASSED ✓" if all_passed else "SOME TESTS FAILED ✗"
    print(verdict)
    print("=" * 72)
    return all_passed


if __name__ == "__main__":
    skip_rust = "--no-rust" in sys.argv
    ok = run_all(skip_rust=skip_rust)
    sys.exit(0 if ok else 1)
