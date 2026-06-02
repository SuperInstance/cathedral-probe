# cathedral-probe v0.2.0 — Real-World Test Diary

**Tester:** Aisha, ML Engineer (Fintech — Fraud Detection)
**Date:** 2026-06-01
**Repo:** `SuperInstance/cathedral-probe` (cloned 2026-06-01)

---

## 1. First Impression — What Does the README Tell You?

**Verdict: Clear, well-structured, knows its audience.**

The README immediately tells me this is a spectral graph analysis crate for "component graphs" — microservices, dependency graphs, communication networks. It opens with a 30-second example that compiles and runs. That's the right way to sell a numeric library.

The API reference is thorough (`spectrum`, `fiedler_value`, `cheeger_*`, `fragility_index`, `component_importance`, `bottlenecks`). It name-drops Fiedler (1973), Chung (1997), Mohar (1989), and Alon-Milman (1985) — real mathematical references, not buzzwords. That gives confidence this isn't a toy.

**What's missing from the README:**
- No mention of the `SparseCathedralProbe` / `DirectedCathedralProbe` types (these exist in the source but the README only shows `CathedralProbe`)
- No mention of `effective_resistance`, `kirchhoff_index`, `spectral_embedding`, `spectral_cluster`, `community_profile`, `fiedler_sensitivity`, `condition_number` — these are all implemented but undocumented in the README
- No performance numbers / benchmark section
- No mention of the `serde` feature
- No "what size graphs can I use this on" guidance

**Score for README: 3.5/5** — good for what it shows, incomplete for what the crate actually does.

---

## 2. Installation — Does `cargo build` Work?

```bash
$ cargo build
Compiling cathedral-probe v0.2.0 (/tmp/beta-cathedral-deepseek)
Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.48s
```

**Zero issues.** Clean compile, no warnings. The crate has two optional dependencies (serde + serde_json) with no required runtime deps.

```bash
$ cargo test
running 115 tests
test result: ok. 115 passed; 0 failed
```

**115 tests, all pass.** That's a lot of tests for a 1700-line crate. Impressive coverage. Tests include:
- Exact eigenvalue verification for K_n, P_n, C_n, S_n (known mathematical results)
- Sparse vs dense agreement
- Directed graph Laplacian tests
- Effective resistance Kirchhoff index verification
- Eigenvector orthogonality and eigenvalue equation satisfaction
- Spectral clustering on two-community graphs
- Serde round-trips (feature-gated)
- Matrix constructor error cases
- Fiedler sensitivity

**Installation score: 5/5** — flawless.

---

## 3. API Exploration — Source Code Deep Dive

I read the full 1700 lines of `src/lib.rs`. Here's my analysis:

### What's Public

| Module | Struct | Visibility | Description |
|--------|--------|-----------|-------------|
| Core | `CathedralProbe` | pub | Dense undirected graph (≤~200 nodes) |
| Core | `SparseCathedralProbe` | pub | CSR sparse graph via Lanczos (100+ nodes) |
| Core | `DirectedCathedralProbe` | pub | Chung's directed Laplacian |
| Results | `SpectrumResult` | pub | Eigenvalues + diagnostics |
| Results | `ComponentAnalysis` | pub | Per-component spectral metrics |
| Results | `SpectrumMethod` | pub | DenseImplicitQr / Lanczos enum |
| Errors | `CathedralError` | pub | NodeNotFound, InsufficientNodes, etc. |

### What's NOT Public (but should be?)

- `fn full_eigen()` — returns eigenvectors but is `fn`-only (not `pub fn`). Actually wait, looking carefully:

```
fn full_eigen(&self) -> (Vec<f64>, Vec<Vec<f64>>) {
```

This is **module-private**, not `pub`. But `effective_resistance`, `kirchhoff_index`, `spectral_embedding`, `spectral_cluster`, `community_profile`, and `fiedler_sensitivity` all call it internally. So users **cannot access eigenvectors directly**.

**Production issue:** If I can't get the Fiedler vector (the eigenvector corresponding to λ₂), I can't implement things like:
- Custom spectral clustering with different k
- Normalized cut computation
- Spectral drawing/layout
- Custom threshold-based anomaly detection
- Data integration (embed transactions as coordinates)

The `full_eigen` method not being public is a **gating issue** for production use.

### Documentation Quality

All public methods have doc comments. Internal methods do not. The numerical algorithm comments cite Golub & Van Loan and mention convergence properties. That's good.

### Code Quality Observations

**The good:**
- `#![deny(unsafe_code)]` — zero unsafe blocks
- Clean separation of concerns
- Householder tridiagonalization + implicit QR with Wilkinson shifts — legitimate numerical linear algebra
- Lanczos with full reorthogonalization — correct for numerical stability
- Eigenvalue equation verified in tests (`L * v = λ * v`)
- Givens rotation utility correct

**The less good:**
- `fiedler_value()` for disconnected graphs returns 0 (and thus `component_importance()` and `bottlenecks()` return zeros). This is **mathematically correct** but **practically useless** for disconnected graphs. You have to use `per_component_analysis()` to get meaningful results.
- `spectral_embedding` sorts by eigenvalue but the `community_profile` and `fiedler_sensitivity` just grab `vecs[1]` — if the nodes list doesn't match internally, you'd get wrong results. (This works because they share the same node ordering, but it's fragile.)
- The Lanczos implementation uses a hardcoded seed (`12345`) for the initial vector — fine for reproducibility but not cryptographically random.

---

## 4. Build a Real Example — Fraud Detection on Transaction Networks

I built `examples/fraud_detection.rs` simulating a fintech transaction graph with:
- **10 normal accounts** (chain-like P2P transfers, weight ~0.3–1.0)
- **5 suspicious accounts** (dense complete subgraph, weight ~8–13 — money laundering layering)
- **5 mule accounts** (receive from suspicious, forward to external)
- **1 external gateway** (cash-out node)
- **3 dormant accounts** (no edges at all)
- **1 thin bridge** (Jack ↔ Sarah, weight=0.1)

### What Worked

| Feature | Result | Useful? |
|---------|--------|---------|
| Global Fiedler | 0.0 (disconnected) | ✅ Mathematically correct |
| Per-component analysis | 4 components, main graph Fiedler=0.0129 | ✅ Correct — bridged network |
| Spectral clustering (k=3) | Cluster 2 = suspicious + mules + gateway | ✅ **Excellent** — isolates fraud ring |
| Effective resistance (within suspicious) | Sarah↔Wu = 0.0322 (very LOW) | ✅ High cohesion detected |
| Effective resistance (cross-community) | Alice↔Sarah = 19.64 (very HIGH) | ✅ Fraud ring isolated |
| Effective resistance (mule network) | Sarah↔Mule = 0.089 | ✅ Mule path short |
| Condition number | κ = 5218 (extremely high) | ✅ Correct — near-disconnected |
| Kirchhoff index | 2273 (very large) | ✅ Confirms fragility |
| Community profile | conductance dips at size=10 (0.008) | ✅ Tight community detected |

### What Failed / Misleading

| Feature | Result | Problem |
|---------|--------|---------|
| `component_importance()` | All zeros | ⚠️ Algorithm uses global Fiedler (which is 0 for disconnected graphs) |
| `bottlenecks()` | All zeros | ⚠️ Same root cause — base Fiedler is 0 |
| `fiedler_sensitivity()` | All zeros | ⚠️ Same root cause |
| `is_healthy(1.0)` | `false` | ⚠️ Technically correct, but not actionable |
| Dormant account resistance | Dormant1↔Dormant2 = 0.0 | ⚠️ This should be INFINITY — disconnected nodes show 0 effective resistance |

### Key Discoveries

**The disconnected-graph bug in importance/bottlenecks/sensitivity:** These methods compute `base_fiedler = self.fiedler_value()` which is 0 when the graph has multiple connected components. Since all edges have *zero* impact on a Fiedler value that's already zero, every importance score is 0.0. This makes these three methods **effectively broken for production graphs** unless you manually call `per_component_analysis()` first and subdivide your graph.

**The dormant-resistance bug:** `effective_resistance()` for disconnected nodes skips divisions by zero eigenvalues but doesn't account for the fact that resistance should be infinite when there's no path between nodes. It returns a finite value that's misleading.

---

## 5. What's MISSING for Production?

For my fintech fraud detection pipeline, here's what cathedral-probe needs before I can deploy it:

### 🔴 Blockers (won't use without)

1. **Public Fiedler vector access** — `full_eigen()` is private. I need the actual eigenvectors to do custom scoring, embedding, and integration with downstream ML models.

2. **Disconnected-graph robustness** — `component_importance()`, `bottlenecks()`, and `fiedler_sensitivity()` give all zeros for disconnected graphs. These should either:
   - Raise an error explaining the graph is disconnected, or
   - Fall back to per-component analysis internally
   
3. **Effective resistance for disconnected nodes** — Should return `f64::INFINITY` for nodes in different connected components.

### 🟡 Strong Wants

4. **Normalized Laplacian** — For degree-skewed transaction graphs, the normalized Laplacian (Chung 1997) handles scale better than the unnormalized version. E.g., a hub account vs a normal account.

5. **PageRank-style importance** — `component_importance` measures Fiedler drop, but for fraud I also want something like personalized PageRank or heat kernel.

6. **Lanczos timeout/fallback** — For larger graphs, Lanczos can fail to converge. The current implementation doesn't have a graceful degradation strategy.

7. **Performance benchmarks** — No guidance on when to use dense vs sparse. For a 100-node fraud graph, the dense solver is fine, but what about 10k nodes?

### 🟢 Nice-to-Haves

8. **Serde by default** — For a monitoring pipeline, I need to serialize `SpectrumResult` and `ComponentAnalysis`. This is currently behind a feature flag.

9. **Parallel matrix construction** — `build_laplacian` uses nested Vec. For larger graphs, a rayon-parallelized version would help.

10. **Dot/Graphviz export** — Would be nice for visualizing community structures.

11. **Benchmarks** — The Cargo.toml has `[dev-dependencies]` with nothing. Add `criterion` benches.

12. **Async/streaming API** — For my real-time transaction monitoring, I'd want to incrementally update eigenvalues as new edges arrive.

---

## 6. Score: 2.5 / 5 Stars

### Breakdown

| Category | Score | Reasoning |
|----------|-------|-----------|
| README clarity | 3.5/5 | Good overview, but hides advanced features |
| Installation / Build | 5/5 | Flawless. Zero deps, fast compile, 115 passing tests |
| API Design | 3/5 | Clean surface area, but `full_eigen()` being private is a blocker |
| Numerical accuracy | 4.5/5 | Tests verify against known eigenvalues. Householder+QR is legitimate |
| Production readiness | 1/5 | Three methods broken for disconnected graphs, no eigenvector access |
| Documentation | 3/5 | Good doc comments in code, but README incomplete re: available features |

**Overall: 2.5/5**

This is an **impressive piece of work** for v0.2.0. The numerical foundation is solid — Householder tridiagonalization + implicit QR Wilkinson shifts is the real deal. The test suite is genuinely excellent (115 tests, many verifying against known exact eigenvalues).

But as a production tool for fraud detection, the disconnected-graph blind spot and private eigenvector access are showstoppers. **For anything with multiple components or isolated nodes — which is basically every real-world graph — the component importance, bottleneck, and sensitivity methods give you cargo-cult zeros.**

### What Would Make It 5 Stars

1. **Make `full_eigen()` public** — or provide a `fiedler_vector()` method returning `Vec<f64>`.
2. **Fix importance/bottlenecks/sensitivity for disconnected graphs** — auto-detect, subdivide by component, or error with guidance.
3. **Fix effective resistance for disconnected nodes** — return `f64::INFINITY`.
4. **Update the README** — document all public types and methods, not just the basic ones.
5. **Add normalized Laplacian support** — critical for degree-skewed real-world graphs.
6. **Document performance characteristics** — "this crate handles up to N nodes in M seconds."

For a v0.3 with those fixes, I'd rate it **4/5**. For 5/5, add benchmarks, serde by default, and a real-world example in the examples directory.

---

## Appendix: Issues Logged (for the repo)

| # | Severity | File | Description |
|---|----------|------|-------------|
| 1 | Major | `lib.rs:full_eigen()` | Private method — blocks access to eigenvectors |
| 2 | Major | `lib.rs:component_importance()` | Returns all zeros when graph has multiple connected components |
| 3 | Major | `lib.rs:bottlenecks()` | Same issue — disconnected graph → all zeros |
| 4 | Major | `lib.rs:fiedler_sensitivity()` | Same issue — zero sensitivity for all edges |
| 5 | Medium | `lib.rs:effective_resistance()` | Returns finite value for nodes in different components (should be INF) |
| 6 | Minor | `README.md` | Doesn't document ~50% of the public API |
| 7 | Minor | `lanczos_eigenvalues()` | Hardcoded RNG seed |
| 8 | Enhancement | `Cargo.toml` | Add `criterion` for benchmarks, add `rayon` optional |
