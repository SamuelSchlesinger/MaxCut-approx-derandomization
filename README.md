# Derandomization: MaxCut Algorithms

Companion code for the derandomization video. Implements three MaxCut algorithms in both Python and Rust:

1. **Randomized** — independent fair coin per vertex; E[cut] = |E|/2
2. **Conditional Expectations** (Vadhan Alg. 3.17) — greedy deterministic; cut ≥ |E|/2
3. **Pairwise Independence** (Vadhan Alg. 3.20) — XOR construction with k = ⌈log₂(n+1)⌉ seed bits; cut ≥ |E|/2

Reference: Vadhan, *Pseudorandomness*, FnTTCS Vol. 7, 2012.

## Usage

**Python**
```
python python/maxcut.py
```

**Rust**
```
cargo run --manifest-path rust/Cargo.toml --release
```

Graph input format (stdin):
```
n m
u_1 v_1
...
u_m v_m
```

Example:
```
echo "4 6
0 1
0 2
0 3
1 2
1 3
2 3" | ./rust/target/release/maxcut
```
