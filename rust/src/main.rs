//! MaxCut CLI binary.
//!
//! Reads a graph from stdin in the format:
//!     n m
//!     u_1 v_1
//!     u_2 v_2
//!     ...
//!     u_m v_m
//!
//! Prints results for all three algorithms in "key: value" format
//! (parseable by verify.py for cross-validation).
//!
//! Usage:
//!     echo "4 6
//!     0 1
//!     0 2
//!     0 3
//!     1 2
//!     1 3
//!     2 3" | ./target/release/maxcut

use std::io::{self, BufRead};
use maxcut::{
    conditional_expectations_maxcut, pairwise_independent_maxcut, randomized_maxcut,
    seed_bits_needed, Edge,
};

// ---------------------------------------------------------------------------
// Simple deterministic PRNG for the randomized algorithm demo
// (xorshift64 — no external dependencies needed)
// ---------------------------------------------------------------------------

struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        // Ensure non-zero state
        Self(if seed == 0 { 0xdeadbeef } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn next_bit(&mut self) -> u8 {
        (self.next_u64() & 1) as u8
    }
}

// ---------------------------------------------------------------------------
// I/O
// ---------------------------------------------------------------------------

fn read_graph() -> (usize, Vec<Edge>) {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines().map(|l| l.expect("read error"));

    let header = lines.next().expect("expected 'n m' header");
    let mut parts = header.split_whitespace();
    let n: usize = parts.next().expect("n").parse().expect("n not integer");
    let m: usize = parts.next().expect("m").parse().expect("m not integer");

    let mut edges = Vec::with_capacity(m);
    for _ in 0..m {
        let line = lines.next().expect("expected edge line");
        let mut p = line.split_whitespace();
        let u: usize = p.next().expect("u").parse().expect("u not integer");
        let v: usize = p.next().expect("v").parse().expect("v not integer");
        edges.push((u, v));
    }

    (n, edges)
}

fn fmt_assignment(a: &[u8]) -> String {
    a.iter()
        .map(|b| b.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let (n, edges) = read_graph();
    let m = edges.len();
    let k = seed_bits_needed(n);

    // --- Randomized (seed 12345) ---
    let mut rng = XorShift64::new(12345);
    let coins: Vec<u8> = (0..n).map(|_| rng.next_bit()).collect();
    let (rand_assignment, rand_cut) = randomized_maxcut(n, &edges, &coins);

    // --- Conditional Expectations ---
    let (ce_assignment, ce_cut) = conditional_expectations_maxcut(n, &edges);

    // --- Pairwise Independence ---
    let (pi_assignment, pi_cut) = pairwise_independent_maxcut(n, &edges);

    // Output (key: value format, parseable by verify.py)
    println!("n: {n}");
    println!("m: {m}");
    println!("half_e: {}", m as f64 / 2.0);
    println!("k: {k}");
    println!("num_seeds: {}", 1usize << k);
    println!();
    println!("rand_cut: {rand_cut}");
    println!("rand_assignment: {}", fmt_assignment(&rand_assignment));
    println!();
    println!("ce_cut: {ce_cut}");
    println!("ce_assignment: {}", fmt_assignment(&ce_assignment));
    println!();
    println!("pi_cut: {pi_cut}");
    println!("pi_assignment: {}", fmt_assignment(&pi_assignment));
    println!();

    // Summary
    let ce_ok = 2 * ce_cut >= m;
    let pi_ok = 2 * pi_cut >= m;
    println!("guarantee_ce: {}", if ce_ok { "PASS" } else { "FAIL" });
    println!("guarantee_pi: {}", if pi_ok { "PASS" } else { "FAIL" });
}
