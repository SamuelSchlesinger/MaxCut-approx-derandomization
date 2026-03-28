"""
maxcut.py — MaxCut algorithms for the Derandomization video.

Three algorithms:
    1. randomized_maxcut
       Flip a coin per vertex. E[cut] = |E|/2 >= OPT/2 (1/2-approx in expectation).

    2. conditional_expectations_maxcut  (Vadhan Algorithm 3.17)
       Deterministic greedy. Processes vertices in order, always placing each
       vertex on the OPPOSITE side from its majority of already-placed neighbors.
       Guarantees: cut >= |E|/2.

    3. pairwise_independent_maxcut  (Vadhan Algorithm 3.20)
       Build n pairwise-independent bits from k = ceil(log2(n+1)) seed bits via
       the XOR construction (Vadhan Construction 3.18). Enumerate all 2^k = O(n)
       seeds; return the best cut found.
       Guarantees: cut >= |E|/2.

References:
    Vadhan, "Pseudorandomness," Foundations and Trends in Theoretical CS,
    Vol. 7, Nos. 1-3, 2012.
    Chapter 3: https://people.seas.harvard.edu/~salil/cs225/spring09/lecnotes/Chap3.pdf
"""

import math
import random
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Edge = Tuple[int, int]  # (u, v), 0-indexed, undirected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cut_size(n: int, edges: List[Edge], assignment: List[int]) -> int:
    """Number of edges with endpoints on opposite sides. assignment[v] in {0, 1}."""
    return sum(1 for u, v in edges if assignment[u] != assignment[v])


def seed_bits_needed(n: int) -> int:
    """
    Minimum k such that 2^k - 1 >= n, i.e. k = ceil(log2(n+1)).

    With k seed bits we can generate 2^k - 1 pairwise-independent bits.
    """
    if n == 0:
        return 0
    return math.ceil(math.log2(n + 1))


def nonempty_subsets(k: int) -> List[int]:
    """All 2^k - 1 nonempty subset bitmasks of {0,...,k-1}, in order 1 .. 2^k-1."""
    return list(range(1, 1 << k))


def pairwise_independent_bits(seed: int, k: int, n: int) -> List[int]:
    """
    Generate n pairwise-independent unbiased bits from a k-bit seed.

    Construction 3.18 (Vadhan):
        Let B_0, ..., B_{k-1} be the bits of `seed`.
        For each nonempty S ⊆ {0,...,k-1}, define R_S = XOR_{i in S} B_i.
        Enumerate nonempty subsets in bitmask order 1, 2, 3, ... and assign
        the i-th subset (0-indexed) to vertex i.

    Requires n <= 2^k - 1.

    Args:
        seed: integer in [0, 2^k), whose bits are B_0,...,B_{k-1}
        k:    number of seed bits
        n:    number of bits to generate

    Returns:
        list of n bits in {0, 1}
    """
    subsets = nonempty_subsets(k)
    if n > len(subsets):
        raise ValueError(
            f"Need n <= 2^k - 1 = {len(subsets)}, got n={n} with k={k}. "
            f"Use k >= {seed_bits_needed(n)}."
        )
    result = []
    for i in range(n):
        mask = subsets[i]
        bit = 0
        for j in range(k):
            if mask & (1 << j):
                bit ^= (seed >> j) & 1
        result.append(bit)
    return result


# ---------------------------------------------------------------------------
# Algorithm 1: Randomized MaxCut
# ---------------------------------------------------------------------------

def randomized_maxcut(
    n: int,
    edges: List[Edge],
    coins: Optional[List[int]] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[List[int], int]:
    """
    Randomized MaxCut.

    Assign each vertex to side 0 (S) or 1 (T) by an independent fair coin flip.

    E[|cut|] = sum_{(u,v) in E} Pr[coin_u != coin_v] = |E|/2.
    Since OPT <= |E|, this is a 1/2-approximation in expectation.

    Args:
        n:     number of vertices (0-indexed)
        edges: list of (u, v) pairs
        coins: optional explicit bit vector of length n (overrides rng)
        rng:   optional seeded random.Random (used when coins=None)

    Returns:
        (assignment, cut_size)
    """
    if coins is not None:
        if len(coins) != n:
            raise ValueError(f"Expected {n} coins, got {len(coins)}")
        assignment = list(coins)
    else:
        r = rng if rng is not None else random.Random()
        assignment = [r.randint(0, 1) for _ in range(n)]
    return assignment, cut_size(n, edges, assignment)


# ---------------------------------------------------------------------------
# Algorithm 2: Conditional Expectations (Vadhan Algorithm 3.17)
# ---------------------------------------------------------------------------

def conditional_expectations_maxcut(
    n: int, edges: List[Edge]
) -> Tuple[List[int], int]:
    """
    Deterministic MaxCut via the Method of Conditional Expectations.

    Define e(r_1,...,r_i) = conditional expected cut size given the first i
    vertex assignments.  By Vadhan eq. 3.2:

        e(r_1,...,r_i) = |cut(S_i, T_i)|
                       + (1/2) * |{edges with >= 1 endpoint in U_i}|

    where S_i, T_i are placed vertices and U_i = {i+1,...,n-1} is undecided.

    The greedy decision for vertex i+1 compares e(..., 0) vs e(..., 1).
    The difference simplifies to:

        e(..., r_{i+1}=0) - e(..., r_{i+1}=1)
            = |cut({i+1}, T_i)| - |cut({i+1}, S_i)|
            = (# placed neighbors in T) - (# placed neighbors in S)

    So: place vertex i in T (side 1) iff it has >= as many placed neighbors in S
    as in T (tie-breaks to T).  Equivalently: place each vertex on the OPPOSITE
    side from its majority of already-placed neighbors.

    Guarantees: cut >= |E|/2.

    Returns:
        (assignment, cut_size)
    """
    assignment: List[int] = [-1] * n  # -1 = unassigned sentinel
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for i in range(n):
        neighbors_in_S = sum(1 for j in adj[i] if assignment[j] == 0)
        neighbors_in_T = sum(1 for j in adj[i] if assignment[j] == 1)
        # Place in T (1) iff cut({i}, S_i) >= cut({i}, T_i), i.e. S-neighbors >= T-neighbors
        assignment[i] = 1 if neighbors_in_S >= neighbors_in_T else 0

    return assignment, cut_size(n, edges, assignment)


# ---------------------------------------------------------------------------
# Algorithm 3: Pairwise Independence (Vadhan Algorithm 3.20)
# ---------------------------------------------------------------------------

def pairwise_independent_maxcut(
    n: int, edges: List[Edge]
) -> Tuple[List[int], int]:
    """
    Deterministic MaxCut via the Pairwise Independence construction.

    The randomized MaxCut analysis only uses pairwise independence:

        E[|cut|] = sum_{(u,v) in E} Pr[R_u != R_v] = |E|/2

    holds whenever each R_i is unbiased and each pair (R_i, R_j) is independent.

    Using k = ceil(log2(n+1)) seed bits, the XOR construction (Construction 3.18)
    produces 2^k - 1 >= n pairwise-independent bits.  Enumerating all 2^k = O(n)
    seeds and taking the best cut gives a deterministic algorithm with the same
    guarantee.

    Guarantees: cut >= |E|/2.

    Returns:
        (best_assignment, best_cut_size)
    """
    if n == 0:
        return [], 0

    k = seed_bits_needed(n)
    best_assignment: Optional[List[int]] = None
    best_cut = -1

    for seed in range(1 << k):
        assignment = pairwise_independent_bits(seed, k, n)
        c = cut_size(n, edges, assignment)
        if c > best_cut:
            best_cut = c
            best_assignment = assignment[:]

    assert best_assignment is not None
    return best_assignment, best_cut
