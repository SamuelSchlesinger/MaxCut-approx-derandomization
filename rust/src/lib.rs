//! MaxCut algorithms: randomized and two deterministic derandomizations.
//!
//! Three algorithms:
//!   - [`randomized_maxcut`]: explicit coin vector; E[cut] = |E|/2
//!   - [`conditional_expectations_maxcut`]: deterministic greedy; cut >= |E|/2
//!   - [`pairwise_independent_maxcut`]: enumerate O(n) seeds; cut >= |E|/2
//!
//! References:
//!   Vadhan, "Pseudorandomness," Foundations and Trends in Theoretical CS,
//!   Vol. 7, Nos. 1–3, 2012.  Chapter 3:
//!   <https://people.seas.harvard.edu/~salil/cs225/spring09/lecnotes/Chap3.pdf>

/// An undirected edge, 0-indexed vertices.
pub type Edge = (usize, usize);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Number of edges with endpoints on opposite sides.
/// `assignment[v]` must be 0 or 1.
pub fn cut_size(edges: &[Edge], assignment: &[u8]) -> usize {
    edges
        .iter()
        .filter(|&&(u, v)| assignment[u] != assignment[v])
        .count()
}

/// Minimum k such that 2^k − 1 >= n, i.e. k = ceil(log2(n + 1)).
///
/// With k seed bits the XOR construction generates exactly 2^k − 1
/// pairwise-independent bits.
pub fn seed_bits_needed(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    (n + 1).next_power_of_two().trailing_zeros() as usize
}

/// All 2^k − 1 nonempty subset bitmasks of {0,...,k−1} in order 1..2^k−1.
pub fn nonempty_subsets(k: usize) -> Vec<usize> {
    (1..(1usize << k)).collect()
}

/// Generate `n` pairwise-independent unbiased bits from a `k`-bit seed.
///
/// Construction 3.18 (Vadhan):
///   Let B_0,...,B_{k−1} be the bits of `seed`.
///   For each nonempty S ⊆ {0,...,k−1} define R_S = XOR_{i in S} B_i.
///   Enumerate nonempty subsets in bitmask order 1, 2, 3, … and assign
///   the i-th subset (0-indexed) to vertex i.
///
/// Requires n <= 2^k − 1.
pub fn pairwise_independent_bits(seed: usize, k: usize, n: usize) -> Vec<u8> {
    let subsets = nonempty_subsets(k);
    assert!(
        n <= subsets.len(),
        "Need n <= 2^k - 1 = {}, got n={} with k={}",
        subsets.len(),
        n,
        k
    );
    (0..n)
        .map(|i| {
            let mask = subsets[i];
            let mut bit = 0u8;
            for j in 0..k {
                if mask & (1 << j) != 0 {
                    bit ^= ((seed >> j) & 1) as u8;
                }
            }
            bit
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Algorithm 1: Randomized MaxCut
// ---------------------------------------------------------------------------

/// Randomized MaxCut with an explicit coin vector.
///
/// Assign vertex i to side 0 (S) or 1 (T) according to `coins[i]`.
/// When `coins` consists of n independent fair bits:
///     E[|cut|] = Σ_{(u,v)} Pr[coin_u ≠ coin_v] = |E|/2.
///
/// `coins` must have length n; elements must be 0 or 1.
pub fn randomized_maxcut(n: usize, edges: &[Edge], coins: &[u8]) -> (Vec<u8>, usize) {
    assert_eq!(coins.len(), n, "coins length must equal n");
    let assignment = coins.to_vec();
    let c = cut_size(edges, &assignment);
    (assignment, c)
}

// ---------------------------------------------------------------------------
// Algorithm 2: Conditional Expectations (Vadhan Algorithm 3.17)
// ---------------------------------------------------------------------------

/// Deterministic MaxCut via the Method of Conditional Expectations.
///
/// Processes vertices 0, 1, ..., n−1 in order.  At step i, define:
///
///   e(r_0,...,r_{i−1}) = |cut(S_i, T_i)| + (1/2)|{edges with ≥1 endpoint in U_i}|
///
/// where S_i, T_i are placed vertices and U_i = {i,...,n−1} is undecided.
///
/// The greedy decision simplifies to (Vadhan eq. 3.2):
///
///   e(..., r_i=0) − e(..., r_i=1) = |cut({i}, S_i)| − |cut({i}, T_i)|
///                                  = (neighbors in S) − (neighbors in T)
///
/// So: place vertex i in T (1) iff it has ≥ as many placed neighbors in S as in T.
/// Equivalently: place each vertex on the OPPOSITE side from its majority of
/// already-placed neighbors.
///
/// Tie-breaking: S-neighbors == T-neighbors → T (side 1).
///
/// Guarantees: cut ≥ |E|/2.
pub fn conditional_expectations_maxcut(n: usize, edges: &[Edge]) -> (Vec<u8>, usize) {
    const UNSET: u8 = 255; // sentinel for unassigned
    let mut assignment = vec![UNSET; n];

    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    for &(u, v) in edges {
        adj[u].push(v);
        adj[v].push(u);
    }

    for i in 0..n {
        let neighbors_in_s = adj[i].iter().filter(|&&j| assignment[j] == 0).count();
        let neighbors_in_t = adj[i].iter().filter(|&&j| assignment[j] == 1).count();
        // Place in T (1) iff S-neighbors >= T-neighbors (tie → T)
        assignment[i] = if neighbors_in_s >= neighbors_in_t { 1 } else { 0 };
    }

    let c = cut_size(edges, &assignment);
    (assignment, c)
}

// ---------------------------------------------------------------------------
// Algorithm 3: Pairwise Independence (Vadhan Algorithm 3.20)
// ---------------------------------------------------------------------------

/// Deterministic MaxCut via the Pairwise Independence construction.
///
/// The randomized MaxCut analysis only uses pairwise independence:
///
///   E[|cut|] = Σ_{(u,v)∈E} Pr[R_u ≠ R_v] = |E|/2
///
/// holds for any pairwise-independent unbiased distribution on (R_0,...,R_{n−1}).
///
/// Using k = ceil(log2(n+1)) seed bits, the XOR construction produces
/// 2^k − 1 ≥ n pairwise-independent bits.  Enumerating all 2^k = O(n) seeds
/// and returning the best cut is a deterministic O(n·m)-time algorithm.
///
/// Tie-breaking: first seed achieving the maximum cut wins.
///
/// Guarantees: cut ≥ |E|/2.
pub fn pairwise_independent_maxcut(n: usize, edges: &[Edge]) -> (Vec<u8>, usize) {
    if n == 0 {
        return (vec![], 0);
    }
    let k = seed_bits_needed(n);
    let num_seeds = 1usize << k;

    let mut best_assignment: Vec<u8> = vec![];
    let mut best_cut: Option<usize> = None;

    for seed in 0..num_seeds {
        let assignment = pairwise_independent_bits(seed, k, n);
        let c = cut_size(edges, &assignment);
        if best_cut.map_or(true, |bc| c > bc) {
            best_cut = Some(c);
            best_assignment = assignment;
        }
    }

    (best_assignment, best_cut.unwrap_or(0))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Graph generators ---

    fn complete_graph(n: usize) -> Vec<Edge> {
        (0..n)
            .flat_map(|u| ((u + 1)..n).map(move |v| (u, v)))
            .collect()
    }

    fn cycle_graph(n: usize) -> Vec<Edge> {
        (0..n).map(|i| (i, (i + 1) % n)).collect()
    }

    fn path_graph(n: usize) -> Vec<Edge> {
        if n < 2 {
            return vec![];
        }
        (0..n - 1).map(|i| (i, i + 1)).collect()
    }

    fn bipartite_complete(a: usize, b: usize) -> (usize, Vec<Edge>) {
        let edges = (0..a)
            .flat_map(|i| (0..b).map(move |j| (i, a + j)))
            .collect();
        (a + b, edges)
    }

    // Tiny deterministic "random" graph: edges (i, (i*7+3) mod n) for i in 0..m.
    fn pseudo_graph(n: usize, m: usize) -> Vec<Edge> {
        let mut edges = vec![];
        let mut seen = std::collections::HashSet::new();
        let mut i = 0usize;
        while edges.len() < m {
            let u = i % n;
            let v = (i * 7 + 3) % n;
            if u != v {
                let e = if u < v { (u, v) } else { (v, u) };
                if seen.insert(e) {
                    edges.push(e);
                }
            }
            i += 1;
            if i > n * n {
                break;
            }
        }
        edges
    }

    // --- Core guarantee: both deterministic algorithms achieve cut >= |E|/2 ---

    fn assert_geq_half(n: usize, edges: &[Edge], label: &str) {
        let m = edges.len();
        let (_, ce) = conditional_expectations_maxcut(n, edges);
        assert!(
            2 * ce >= m,
            "{label}: CE cut {ce} < |E|/2 = {} (m={m})",
            m as f64 / 2.0
        );
        let (_, pi) = pairwise_independent_maxcut(n, edges);
        assert!(
            2 * pi >= m,
            "{label}: PI cut {pi} < |E|/2 = {} (m={m})",
            m as f64 / 2.0
        );
    }

    // --- seed_bits_needed ---

    #[test]
    fn test_seed_bits_needed() {
        // k = ceil(log2(n+1))
        assert_eq!(seed_bits_needed(0), 0); // 2^0 - 1 = 0 >= 0
        assert_eq!(seed_bits_needed(1), 1); // 2^1 - 1 = 1 >= 1
        assert_eq!(seed_bits_needed(2), 2); // 2^2 - 1 = 3 >= 2
        assert_eq!(seed_bits_needed(3), 2); // 2^2 - 1 = 3 >= 3
        assert_eq!(seed_bits_needed(4), 3); // 2^3 - 1 = 7 >= 4
        assert_eq!(seed_bits_needed(7), 3); // 2^3 - 1 = 7 >= 7
        assert_eq!(seed_bits_needed(8), 4); // 2^4 - 1 = 15 >= 8
        assert_eq!(seed_bits_needed(15), 4); // 2^4 - 1 = 15 >= 15
        assert_eq!(seed_bits_needed(16), 5); // 2^5 - 1 = 31 >= 16
        assert_eq!(seed_bits_needed(100), 7); // 2^7 - 1 = 127 >= 100
    }

    // --- pairwise_independent_bits: exact enumeration ---

    #[test]
    fn test_pi_bits_k2_exact() {
        // k=2, n=3: subsets {B0},{B1},{B0,B1}
        // seed 0 (B0=0,B1=0): [0,0,0]
        // seed 1 (B0=1,B1=0): [1,0,1]
        // seed 2 (B0=0,B1=1): [0,1,1]
        // seed 3 (B0=1,B1=1): [1,1,0]
        let expected = vec![
            vec![0u8, 0, 0],
            vec![1, 0, 1],
            vec![0, 1, 1],
            vec![1, 1, 0],
        ];
        for (seed, exp) in expected.iter().enumerate() {
            let got = pairwise_independent_bits(seed, 2, 3);
            assert_eq!(got, *exp, "seed={seed}");
        }
    }

    #[test]
    fn test_pi_bits_k3_first_vertex() {
        // vertex 0 uses subset mask=1={B0}, so bit = B0 = (seed >> 0) & 1
        for seed in 0..8usize {
            let bits = pairwise_independent_bits(seed, 3, 1);
            assert_eq!(bits[0], (seed & 1) as u8, "seed={seed}");
        }
    }

    #[test]
    fn test_pairwise_independence_exact() {
        // For each k in {2,3,4}, check that every pair of generated bits
        // is jointly uniform: each of the 4 values appears exactly 2^(k-2) times.
        for k in 2..=4usize {
            let n = (1 << k) - 1; // use all available bits
            let num_seeds = 1 << k;
            let expected_count = num_seeds / 4;

            for i in 0..n.min(6) {
                for j in (i + 1)..n.min(7) {
                    let mut counts = [0usize; 4];
                    for seed in 0..num_seeds {
                        let bits = pairwise_independent_bits(seed, k, n);
                        let idx = (bits[i] as usize) * 2 + (bits[j] as usize);
                        counts[idx] += 1;
                    }
                    assert_eq!(
                        counts,
                        [expected_count; 4],
                        "k={k} pair ({i},{j}): counts={counts:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_pi_bits_each_unbiased() {
        // Each generated bit should be 0 exactly half the time across all seeds.
        for k in 1..=5usize {
            let n = (1 << k) - 1;
            for i in 0..n {
                let ones: usize = (0..(1 << k))
                    .map(|seed| pairwise_independent_bits(seed, k, n)[i] as usize)
                    .sum();
                assert_eq!(
                    ones,
                    1 << (k - 1),
                    "k={k} vertex {i}: {ones} ones over {} seeds (want {})",
                    1 << k,
                    1 << (k - 1)
                );
            }
        }
    }

    // --- Known-answer tests ---

    #[test]
    fn test_k4_known_answer() {
        // K4: n=4, m=6. Hand-traced CE assignment = [1,0,1,0], cut = 4.
        let n = 4;
        let edges = complete_graph(n);
        let (ce_assign, ce_cut) = conditional_expectations_maxcut(n, &edges);
        assert_eq!(ce_assign, vec![1, 0, 1, 0], "K4 CE assignment");
        assert_eq!(ce_cut, 4, "K4 CE cut");

        let (_, pi_cut) = pairwise_independent_maxcut(n, &edges);
        assert_eq!(pi_cut, 4, "K4 PI cut (optimal)");
    }

    #[test]
    fn test_single_edge() {
        let (ce_assign, ce_cut) = conditional_expectations_maxcut(2, &[(0, 1)]);
        // vertex 0: no neighbors → T(1); vertex 1: neighbor 0 in T → S(0)
        assert_eq!(ce_assign, vec![1, 0], "single edge CE assignment");
        assert_eq!(ce_cut, 1, "single edge CE cut");

        let (_, pi_cut) = pairwise_independent_maxcut(2, &[(0, 1)]);
        assert_eq!(pi_cut, 1, "single edge PI cut");
    }

    #[test]
    fn test_empty_graph() {
        let (assign, c) = conditional_expectations_maxcut(0, &[]);
        assert!(assign.is_empty());
        assert_eq!(c, 0);
        let (assign, c) = pairwise_independent_maxcut(0, &[]);
        assert!(assign.is_empty());
        assert_eq!(c, 0);
    }

    #[test]
    fn test_no_edges() {
        // Graph with vertices but no edges
        let (_, c) = conditional_expectations_maxcut(5, &[]);
        assert_eq!(c, 0);
        let (_, c) = pairwise_independent_maxcut(5, &[]);
        assert_eq!(c, 0);
    }

    #[test]
    fn test_bipartite_optimal() {
        // K_{a,b} with A-vertices 0..a-1 first, B-vertices a..a+b-1 second.
        //
        // CE always finds the OPTIMAL cut = a*b for this vertex ordering:
        //   all A-vertices have no placed neighbors → T; all B-vertices have all
        //   placed neighbors in T → S. Every edge crosses. cut = a*b.
        //
        // PI guarantees cut >= |E|/2 = a*b/2, but does NOT guarantee the optimal.
        // (Example: K_{1,5}: PI achieves 4, not 5. The XOR construction with k=3
        //  cannot produce the all-zeros/all-ones split needed for a perfect cut.)
        for a in 1..=5 {
            for b in 1..=5 {
                let (n, edges) = bipartite_complete(a, b);
                let m = edges.len(); // = a * b
                let (_, ce_cut) = conditional_expectations_maxcut(n, &edges);
                assert_eq!(ce_cut, a * b, "K_{a},{b} CE cut should equal a*b={}", a * b);
                let (_, pi_cut) = pairwise_independent_maxcut(n, &edges);
                assert!(
                    2 * pi_cut >= m,
                    "K_{a},{b} PI cut {pi_cut} < |E|/2 = {}",
                    m as f64 / 2.0
                );
            }
        }
    }

    // --- Guarantee: cut >= |E|/2 on many graphs ---

    #[test]
    fn test_complete_graphs() {
        for n in 1..=10 {
            let edges = complete_graph(n);
            assert_geq_half(n, &edges, &format!("K{n}"));
        }
    }

    #[test]
    fn test_cycle_graphs() {
        for n in 3..=15 {
            let edges = cycle_graph(n);
            assert_geq_half(n, &edges, &format!("C{n}"));
        }
    }

    #[test]
    fn test_path_graphs() {
        for n in 2..=15 {
            let edges = path_graph(n);
            assert_geq_half(n, &edges, &format!("P{n}"));
        }
    }

    #[test]
    fn test_pseudo_random_graphs() {
        let cases = [
            (10, 15),
            (20, 40),
            (30, 80),
            (50, 150),
            (100, 400),
        ];
        for (n, m) in cases {
            let edges = pseudo_graph(n, m);
            assert_geq_half(n, &edges, &format!("pseudo({n},{m})"));
        }
    }

    // --- Randomized algorithm: explicit coins ---

    #[test]
    fn test_randomized_explicit_coins() {
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        // coins [0,1,0,1]: vertices alternate S/T → all 3 edges cut
        let (_, c) = randomized_maxcut(4, &edges, &[0, 1, 0, 1]);
        assert_eq!(c, 3);
        // coins [0,0,0,0]: all in S → cut = 0
        let (_, c) = randomized_maxcut(4, &edges, &[0, 0, 0, 0]);
        assert_eq!(c, 0);
    }

    // --- CE greedy decision rule: hand-verified small cases ---

    #[test]
    fn test_ce_greedy_rule() {
        // Triangle K3: first vertex → T, second → S, third: neighbors in T=1,S=1 → T (tie)
        let n = 3;
        let edges = complete_graph(n);
        let (assign, cut) = conditional_expectations_maxcut(n, &edges);
        assert_eq!(assign, vec![1, 0, 1], "K3 CE assignment");
        // Cut: (0,1)✓ (0,2)✗ (1,2)✓ → 2 out of 3, >= 1.5 ✓
        assert_eq!(cut, 2, "K3 CE cut");
        assert!(2 * cut >= 3, "K3 cut < |E|/2");
    }

    #[test]
    fn test_ce_path_alternating() {
        // Path P_n: CE should alternate T/S (1/0), cutting every edge.
        // vertex 0 → T(1), vertex 1 → S(0), vertex 2 → T(1), ...
        for n in 2..=10 {
            let edges = path_graph(n);
            let (assign, cut) = conditional_expectations_maxcut(n, &edges);
            let expected: Vec<u8> = (0..n)
                .map(|i| if i % 2 == 0 { 1u8 } else { 0u8 })
                .collect();
            assert_eq!(assign, expected, "P{n} CE assignment");
            assert_eq!(cut, n - 1, "P{n} CE cut (all edges cut)");
        }
    }

    // --- PI: best seed gives cut >= |E|/2 even on worst-case-ish graphs ---

    #[test]
    fn test_pi_finds_good_seed() {
        // Verify that among all seeds, at least one gives cut >= |E|/2.
        let graphs: Vec<(&str, usize, Vec<Edge>)> = vec![
            ("K5", 5, complete_graph(5)),
            ("C7", 7, cycle_graph(7)),
            ("K_{3,4}", 7, bipartite_complete(3, 4).1),
        ];
        for (label, n, edges) in graphs {
            let m = edges.len();
            let k = seed_bits_needed(n);
            let mut found = false;
            for seed in 0..(1usize << k) {
                let bits = pairwise_independent_bits(seed, k, n);
                let c = cut_size(&edges, &bits);
                if 2 * c >= m {
                    found = true;
                    break;
                }
            }
            assert!(found, "{label}: no seed achieves cut >= |E|/2");
        }
    }
}
