//! # Cathedral Probe
//!
//! Spectral topology analysis for component graphs.
//!
//! Compute Laplacian eigenvalues, Fiedler value (connectivity), Cheeger constant
//! (bottleneck detection), and component importance. Answer: "is the space between
//! my components healthy?"
//!
//! ## Numerical Methods
//!
//! - **Dense symmetric matrices** (≤~200 nodes): Householder tridiagonalization
//!   followed by implicit QR with Wilkinson shifts (Golub & Van Loan, *Matrix
//!   Computations*, 4th ed., §8.3).
//! - **Sparse matrices** (via `from_edges`): Lanczos iteration with full
//!   reorthogonalization for the top-k eigenvalues.
//! - **Directed graphs**: Chung's directed Laplacian with transition-probability
//!   normalization.

#![deny(unsafe_code)]

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════
// Result types
// ═══════════════════════════════════════════════════════════════════════

/// Method used to compute the spectrum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpectrumMethod {
    /// Householder tridiagonalization + implicit QR with Wilkinson shifts.
    DenseImplicitQr,
    /// Lanczos iteration with full reorthogonalization (top-k eigenvalues).
    Lanczos { k: usize },
}

/// Result of a spectrum computation with diagnostics.
#[derive(Debug, Clone)]
pub struct SpectrumResult {
    /// Eigenvalues sorted in ascending order.
    pub eigenvalues: Vec<f64>,
    /// Number of QR/Lanczos iterations consumed.
    pub iterations: usize,
    /// Whether the algorithm converged within tolerance.
    pub converged: bool,
    /// Which numerical method was used.
    pub method: SpectrumMethod,
}

impl SpectrumResult {
    /// Fiedler value (second-smallest eigenvalue). Returns `None` if < 2 eigenvalues.
    pub fn fiedler_value(&self) -> Option<f64> {
        if self.eigenvalues.len() >= 2 {
            Some(self.eigenvalues[1])
        } else {
            None
        }
    }
}

/// Error type for cathedral-probe operations.
#[derive(Debug, Clone)]
pub enum CathedralError {
    /// Node name not found in the graph.
    NodeNotFound(String),
    /// Insufficient nodes for the requested operation.
    InsufficientNodes { have: usize, need: usize },
    /// The graph is empty.
    EmptyGraph,
    /// Lanczos failed to converge.
    LanczosNoConverge { iterations: usize },
}

impl std::fmt::Display for CathedralError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NodeNotFound(name) => write!(f, "node not found: {name}"),
            Self::InsufficientNodes { have, need } => {
                write!(f, "need {need} nodes but graph has {have}")
            }
            Self::EmptyGraph => write!(f, "graph is empty"),
            Self::LanczosNoConverge { iterations } => {
                write!(f, "Lanczos failed to converge after {iterations} iterations")
            }
        }
    }
}

impl std::error::Error for CathedralError {}

// ═══════════════════════════════════════════════════════════════════════
// Dense undirected graph
// ═══════════════════════════════════════════════════════════════════════

/// A weighted undirected graph of named components.
pub struct CathedralProbe {
    nodes: Vec<String>,
    node_index: HashMap<String, usize>,
    edges: Vec<(usize, usize, f64)>,
}

impl CathedralProbe {
    /// Create a new graph with named components.
    pub fn new(components: Vec<&str>) -> Self {
        let nodes: Vec<String> = components.iter().map(|s| s.to_string()).collect();
        let node_index: HashMap<String, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        Self {
            nodes,
            node_index,
            edges: Vec::new(),
        }
    }

    /// Add a weighted edge between two components.
    pub fn connect(&mut self, a: &str, b: &str, weight: f64) {
        if let (Some(&i), Some(&j)) = (self.node_index.get(a), self.node_index.get(b)) {
            self.edges.push((i, j, weight));
        }
    }

    /// Number of components.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Component names.
    pub fn components(&self) -> &[String] {
        &self.nodes
    }

    // ─── Laplacian ──────────────────────────────────────────────

    fn build_laplacian(&self) -> Vec<Vec<f64>> {
        let n = self.nodes.len();
        let mut lap = vec![vec![0.0f64; n]; n];
        for &(i, j, w) in &self.edges {
            lap[i][i] += w;
            lap[j][j] += w;
            lap[i][j] -= w;
            lap[j][i] -= w;
        }
        lap
    }

    /// Compute all eigenvalues of the graph Laplacian.
    ///
    /// Uses Householder tridiagonalization followed by implicit QR iteration
    /// with Wilkinson shifts. Converges cubically for symmetric matrices.
    ///
    /// # Returns
    ///
    /// A `SpectrumResult` with eigenvalues sorted ascending, convergence info,
    /// and the method used.
    pub fn spectrum_result(&self) -> SpectrumResult {
        let n = self.nodes.len();
        if n == 0 {
            return SpectrumResult {
                eigenvalues: vec![],
                iterations: 0,
                converged: true,
                method: SpectrumMethod::DenseImplicitQr,
            };
        }
        if n == 1 {
            return SpectrumResult {
                eigenvalues: vec![self.build_laplacian()[0][0]],
                iterations: 0,
                converged: true,
                method: SpectrumMethod::DenseImplicitQr,
            };
        }

        // Step 1: Householder tridiagonalization
        let mut diag = vec![0.0f64; n];
        let mut subdiag = vec![0.0f64; n - 1];
        householder_tridiag(&self.build_laplacian(), &mut diag, &mut subdiag);

        // Step 2: Implicit QR with Wilkinson shifts on the tridiagonal form
        let (eigs, iters, converged) = implicit_qr_tridiag(&mut diag, &mut subdiag, 1e-14, n * 30);

        // Clean near-zero eigenvalues (Laplacian should have ≥0 eigenvalues)
        let mut eigs = eigs;
        for e in &mut eigs {
            if (*e).abs() < 1e-10 { *e = 0.0; }
            if *e < 0.0 && (*e).abs() < 1e-10 { *e = 0.0; }
        }
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        SpectrumResult {
            eigenvalues: eigs,
            iterations: iters,
            converged,
            method: SpectrumMethod::DenseImplicitQr,
        }
    }

    /// Compute all eigenvalues (convenience wrapper returning just the values).
    pub fn spectrum(&self) -> Vec<f64> {
        self.spectrum_result().eigenvalues
    }

    /// Fiedler value: second-smallest eigenvalue.
    /// Higher = better connected. Zero = disconnected.
    pub fn fiedler_value(&self) -> f64 {
        let spec = self.spectrum();
        if spec.len() >= 2 { spec[1] } else { 0.0 }
    }

    /// Algebraic connectivity alias for Fiedler value.
    pub fn algebraic_connectivity(&self) -> f64 {
        self.fiedler_value()
    }

    /// Spectral conductance upper bound from Cheeger's inequality.
    ///
    /// Returns √(2·λ₂), satisfying: λ₂/2 ≤ h(G) ≤ √(2·λ₂).
    ///
    /// # References
    /// - Fiedler, M. (1973). "Algebraic connectivity of graphs."
    ///   *Czechoslovak Mathematical Journal*, 23(2), 298-305.
    /// - Chung, F. (1997). *Spectral Graph Theory.* CBMS No. 92, AMS.
    /// - Mohar, B. (1989). "Isoperimetric numbers of graphs."
    ///   *J. Combin. Theory Ser. B*, 47(3), 274-291.
    pub fn cheeger_upper_bound(&self) -> f64 {
        let fiedler = self.fiedler_value();
        if self.nodes.len() <= 1 { return 0.0; }
        (2.0 * fiedler).sqrt()
    }

    /// Lower bound on the Cheeger constant: λ₂/2.
    pub fn cheeger_lower_bound(&self) -> f64 {
        self.fiedler_value() / 2.0
    }

    /// Legacy alias for `cheeger_upper_bound()`.
    #[deprecated(note = "Use cheeger_upper_bound() or cheeger_lower_bound() instead")]
    pub fn cheeger_constant(&self) -> f64 {
        self.cheeger_upper_bound()
    }

    /// Quick health check: is the Fiedler value above a minimum threshold?
    pub fn is_healthy(&self, min_fiedler: f64) -> bool {
        self.fiedler_value() >= min_fiedler
    }

    /// Fragility index: 1 / Fiedler. Higher = more fragile.
    /// Returns f64::INFINITY if disconnected (Fiedler = 0).
    pub fn fragility_index(&self) -> f64 {
        let f = self.fiedler_value();
        if f < 1e-12 { f64::INFINITY } else { 1.0 / f }
    }

    /// Component importance: how much does removing each component
    /// reduce the Fiedler value? Higher = more critical.
    pub fn component_importance(&self) -> HashMap<String, f64> {
        let base_fiedler = self.fiedler_value();
        let mut importance = HashMap::new();
        for (name, &idx) in &self.node_index {
            let mut sub = CathedralProbe::new(
                self.nodes.iter()
                    .enumerate()
                    .filter(|(i, _)| *i != idx)
                    .map(|(_, n)| n.as_str())
                    .collect()
            );
            for &(i, j, w) in &self.edges {
                if i != idx && j != idx {
                    let a_name = &self.nodes[i];
                    let b_name = &self.nodes[j];
                    if sub.node_index.contains_key(a_name) && sub.node_index.contains_key(b_name) {
                        sub.connect(a_name, b_name, w);
                    }
                }
            }
            let drop = base_fiedler - sub.fiedler_value();
            importance.insert(name.clone(), drop.max(0.0));
        }
        importance
    }

    /// Identify bottleneck edges — those whose removal most reduces Fiedler value.
    pub fn bottlenecks(&self) -> Vec<(String, String, f64)> {
        let base = self.fiedler_value();
        let mut results = Vec::new();
        for &(i, j, _) in &self.edges {
            let mut sub = self.clone_shallow();
            sub.edges.retain(|&(ei, ej, _)| !(ei == i && ej == j));
            let drop = base - sub.fiedler_value();
            results.push((self.nodes[i].clone(), self.nodes[j].clone(), drop.max(0.0)));
        }
        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        results
    }

    fn clone_shallow(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            node_index: self.node_index.clone(),
            edges: self.edges.clone(),
        }
    }

    /// Number of connected components (count of zero eigenvalues).
    pub fn connected_components(&self) -> usize {
        self.spectrum().iter().filter(|&&e| e.abs() < 1e-8).count().max(1)
    }

    /// Is the graph fully connected?
    pub fn is_connected(&self) -> bool {
        self.connected_components() <= 1
    }

    /// Total edge weight.
    pub fn total_weight(&self) -> f64 {
        self.edges.iter().map(|&(_, _, w)| w).sum()
    }

    /// Average degree (weighted).
    pub fn average_degree(&self) -> f64 {
        if self.nodes.is_empty() { return 0.0; }
        self.total_weight() * 2.0 / self.nodes.len() as f64
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Sparse undirected graph
// ═══════════════════════════════════════════════════════════════════════

/// Sparse graph stored in CSR (Compressed Sparse Row) format.
///
/// Use this for large graphs (100+ nodes) where the Laplacian is sparse.
/// Eigenvalues are computed via Lanczos iteration.
pub struct SparseCathedralProbe {
    n: usize,
    /// CSR row pointers (length n+1).
    row_ptr: Vec<usize>,
    /// CSR column indices.
    col_ind: Vec<usize>,
    /// CSR values (off-diagonal entries of -L, stored as negative weights).
    values: Vec<f64>,
    /// Diagonal of L (degree weights).
    diag: Vec<f64>,
    node_names: Vec<String>,
}

impl SparseCathedralProbe {
    /// Build a sparse graph from an edge list.
    ///
    /// `num_nodes` is the total number of nodes (indexed 0..num_nodes-1).
    /// `edges` is a list of `(i, j, weight)` tuples for undirected edges.
    pub fn from_edges(num_nodes: usize, edges: &[(usize, usize, f64)]) -> Self {
        let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_nodes];
        let mut diag = vec![0.0f64; num_nodes];

        for &(i, j, w) in edges {
            if i < num_nodes && j < num_nodes && i != j {
                adj[i].push((j, w));
                adj[j].push((i, w));
                diag[i] += w;
                diag[j] += w;
            }
        }

        let mut row_ptr = vec![0usize; num_nodes + 1];
        let mut col_ind = Vec::new();
        let mut values = Vec::new();

        for i in 0..num_nodes {
            // Sort neighbors for deterministic CSR
            adj[i].sort_by_key(|(j, _)| *j);
            // Remove duplicate edges (sum weights)
            let mut deduped: Vec<(usize, f64)> = Vec::new();
            for (j, w) in &adj[i] {
                if let Some(last) = deduped.last_mut() {
                    if last.0 == *j {
                        last.1 += w;
                        continue;
                    }
                }
                deduped.push((*j, *w));
            }
            row_ptr[i + 1] = row_ptr[i] + deduped.len();
            for (j, w) in deduped {
                col_ind.push(j);
                values.push(-w); // off-diagonal of L is -w
            }
        }

        let node_names = (0..num_nodes).map(|i| format!("n{i}")).collect();

        Self { n: num_nodes, row_ptr, col_ind, values, diag, node_names }
    }

    /// Assign names to nodes.
    pub fn with_names(mut self, names: Vec<String>) -> Self {
        assert_eq!(names.len(), self.n, "names must have exactly {} elements", self.n);
        self.node_names = names;
        self
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize { self.n }

    /// Compute top-k smallest eigenvalues using Lanczos iteration.
    ///
    /// Uses full reorthogonalization for numerical stability.
    /// Returns a `SpectrumResult` with the k smallest eigenvalues.
    pub fn spectrum_top_k(&self, k: usize) -> Result<SpectrumResult, CathedralError> {
        if self.n == 0 {
            return Ok(SpectrumResult {
                eigenvalues: vec![],
                iterations: 0,
                converged: true,
                method: SpectrumMethod::Lanczos { k },
            });
        }
        let k = k.min(self.n);
        let (eigs, iters, converged) = lanczos_eigenvalues(self, k, self.n * 20, 1e-12);
        Ok(SpectrumResult {
            eigenvalues: eigs,
            iterations: iters,
            converged,
            method: SpectrumMethod::Lanczos { k },
        })
    }

    /// Convenience: compute Fiedler value (second-smallest eigenvalue).
    pub fn fiedler_value(&self) -> f64 {
        if self.n < 2 { return 0.0; }
        self.spectrum_top_k(2).map(|r| r.eigenvalues[1]).unwrap_or(0.0)
    }

    /// Matrix-vector product: y = Lx where L is the graph Laplacian.
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        for i in 0..self.n {
            let mut sum = self.diag[i] * x[i];
            for idx in self.row_ptr[i]..self.row_ptr[i + 1] {
                sum += self.values[idx] * x[self.col_ind[idx]]; // values[idx] = -w
            }
            y[i] = sum;
        }
    }

    /// Total edge weight.
    pub fn total_weight(&self) -> f64 {
        // Each undirected edge is stored once per endpoint in diag, so total = sum(diag)/2
        self.diag.iter().sum::<f64>() / 2.0
    }
}

/// Build a `SparseCathedralProbe` from a `CathedralProbe` (for testing convenience).
impl From<&CathedralProbe> for SparseCathedralProbe {
    fn from(cp: &CathedralProbe) -> Self {
        let edges: Vec<(usize, usize, f64)> = cp.edges.iter().map(|&(i, j, w)| (i, j, w)).collect();
        let mut sparse = Self::from_edges(cp.node_count(), &edges);
        sparse.node_names = cp.nodes.clone();
        sparse
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Directed graph (Chung's directed Laplacian)
// ═══════════════════════════════════════════════════════════════════════

/// A weighted directed graph using Chung's directed Laplacian.
///
/// The directed Laplacian is defined as L = I - (Φ^{1/2} P Φ^{-1/2} + Φ^{-1/2} P^T Φ^{1/2}) / 2,
/// where P is the transition probability matrix and Φ is the stationary distribution.
///
/// For strongly connected directed graphs, the Fiedler value measures
/// how well-connected the graph is under random walks.
///
/// # References
/// - Chung, F. (2005). "Laplacians and the Cheeger inequality for directed graphs."
///   *Annals of Combinatorics*, 9, 1-19.
pub struct DirectedCathedralProbe {
    n: usize,
    /// Out-adjacency: out_edges[i] = vec of (j, weight)
    out_edges: Vec<Vec<(usize, f64)>>,
    /// Stationary distribution φ (computed by PageRank-style iteration).
    phi: Vec<f64>,
    node_names: Vec<String>,
}

impl DirectedCathedralProbe {
    /// Create a directed graph from named components.
    pub fn new(components: Vec<&str>) -> Self {
        let n = components.len();
        Self {
            n,
            out_edges: vec![Vec::new(); n],
            phi: vec![1.0 / n as f64; n],
            node_names: components.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Add a directed edge from `a` to `b` with the given weight.
    pub fn add_edge(&mut self, a: &str, b: &str, weight: f64) {
        if let (Some(i), Some(j)) = (
            self.node_names.iter().position(|n| n == a),
            self.node_names.iter().position(|n| n == b),
        ) {
            self.out_edges[i].push((j, weight));
        }
    }

    /// Compute the stationary distribution φ via power iteration.
    ///
    /// Solves φ^T P = φ^T where P is the row-stochastic transition matrix.
    /// Uses teleportation (α = 0.15) for robustness with dangling nodes.
    #[allow(clippy::needless_range_loop)]
    fn compute_stationary(&mut self) {
        let n = self.n;
        if n == 0 { return; }

        // Build row-stochastic transition matrix
        let mut p = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            let total: f64 = self.out_edges[i].iter().map(|&(_, w)| w).sum();
            if total > 0.0 {
                for &(j, w) in &self.out_edges[i] {
                    p[i][j] = w / total;
                }
            } else {
                // Dangling node: uniform distribution
                for val in p[i].iter_mut().take(n) { *val = 1.0 / n as f64; }
            }
        }

        // Power iteration with teleportation: P_α = (1-α)P + α * (1/n) * 11^T
        let alpha = 0.15;
        let uniform = 1.0 / n as f64;
        let mut phi = vec![uniform; n];

        for _ in 0..200 {
            let mut new_phi = vec![0.0; n];
            for j in 0..n {
                for i in 0..n {
                    new_phi[j] += phi[i] * ((1.0 - alpha) * p[i][j] + alpha * uniform);
                }
            }
            // Normalize
            let sum: f64 = new_phi.iter().sum();
            if sum > 0.0 {
                for v in &mut new_phi { *v /= sum; }
            }
            phi = new_phi;
        }
        self.phi = phi;
    }

    /// Build Chung's directed Laplacian as a dense matrix.
    #[allow(clippy::needless_range_loop)]
    fn build_directed_laplacian(&mut self) -> Vec<Vec<f64>> {
        self.compute_stationary();
        let n = self.n;
        let phi = &self.phi;

        // Build transition matrix P
        let mut p = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            let total: f64 = self.out_edges[i].iter().map(|&(_, w)| w).sum();
            if total > 0.0 {
                for &(j, w) in &self.out_edges[i] {
                    p[i][j] = w / total;
                }
            }
        }

        // Chung's Laplacian: L = I - (Φ^{1/2} P Φ^{-1/2} + Φ^{-1/2} P^T Φ^{1/2}) / 2
        let mut lap = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                let phi_i = phi[i].max(1e-15);
                let phi_j = phi[j].max(1e-15);
                let sym = (phi_i.sqrt() / phi_j.sqrt()) * p[i][j]
                        + (phi_j.sqrt() / phi_i.sqrt()) * p[j][i];
                lap[i][j] = -sym / 2.0;
            }
            lap[i][i] += 1.0;
        }
        lap
    }

    /// Compute the spectrum of the directed Laplacian.
    ///
    /// Returns a `SpectrumResult` with eigenvalues sorted ascending.
    /// The Fiedler value (second smallest) measures directed connectivity.
    pub fn spectrum_result(&mut self) -> SpectrumResult {
        if self.n == 0 {
            return SpectrumResult {
                eigenvalues: vec![],
                iterations: 0,
                converged: true,
                method: SpectrumMethod::DenseImplicitQr,
            };
        }
        if self.n == 1 {
            return SpectrumResult {
                eigenvalues: vec![1.0],
                iterations: 0,
                converged: true,
                method: SpectrumMethod::DenseImplicitQr,
            };
        }

        let lap = self.build_directed_laplacian();
        let mut diag = vec![0.0f64; self.n];
        let mut subdiag = vec![0.0f64; self.n - 1];
        householder_tridiag(&lap, &mut diag, &mut subdiag);

        let (eigs, iters, converged) = implicit_qr_tridiag(&mut diag, &mut subdiag, 1e-12, self.n * 30);

        let mut eigs = eigs;
        for e in &mut eigs {
            if (*e).abs() < 1e-10 { *e = 0.0; }
        }
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        SpectrumResult {
            eigenvalues: eigs,
            iterations: iters,
            converged,
            method: SpectrumMethod::DenseImplicitQr,
        }
    }

    /// Directed Fiedler value.
    pub fn fiedler_value(&mut self) -> f64 {
        let spec = self.spectrum_result();
        if spec.eigenvalues.len() >= 2 { spec.eigenvalues[1] } else { 0.0 }
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize { self.n }
}

// ═══════════════════════════════════════════════════════════════════════
// Householder tridiagonalization
// ═══════════════════════════════════════════════════════════════════════

/// Reduce a symmetric matrix to tridiagonal form via Householder reflections.
///
/// Given symmetric A, computes T = Q^T A Q where T is tridiagonal.
/// Only the diagonal and sub/super-diagonal are extracted.
///
/// Reference: Golub & Van Loan, *Matrix Computations*, 4th ed., Algorithm 8.3.1.
#[allow(clippy::needless_range_loop)]
fn householder_tridiag(a: &[Vec<f64>], diag: &mut [f64], subdiag: &mut [f64]) {
    let n = a.len();
    // Work on a copy
    let mut t = a.to_vec();

    for k in 0..n.saturating_sub(2) {
        // Extract column below the subdiagonal
        let m = n - k - 1;
        let mut x = vec![0.0; m];
        for i in 0..m {
            x[i] = t[k + 1 + i][k];
        }

        // Compute Householder vector
        let _sigma: f64 = x[1..].iter().map(|&v| v * v).sum();
        let alpha = x[0].signum() * x.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-300);
        x[0] += alpha;

        // Normalize v
        let v_norm: f64 = x.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if v_norm < 1e-15 { continue; }
        for v in &mut x { *v /= v_norm; }

        // Apply similarity transformation: T = (I - 2vv^T) T (I - 2vv^T)
        // First: T = (I - 2vv^T) T  → update rows k+1..n
        // For symmetric T, we can use the formula:
        //   p = 2 * T[k+1:, k+1:] * v
        //   q = p - (v^T p) v  (simplified for rank-2 update)
        //   T[k+1:, k+1:] -= v q^T + q v^T

        // But let's do it the straightforward way:
        // Apply from left: T[k+1:, :] -= 2 * v * (v^T * T[k+1:, :])
        for j in 0..n {
            let dot: f64 = (0..m).map(|i| x[i] * t[k + 1 + i][j]).sum();
            for i in 0..m {
                t[k + 1 + i][j] -= 2.0 * x[i] * dot;
            }
        }
        // Apply from right: T[:, k+1:] -= 2 * (T[:, k+1:] * v) * v^T
        for i in 0..n {
            let dot: f64 = (0..m).map(|l| t[i][k + 1 + l] * x[l]).sum();
            for l in 0..m {
                t[i][k + 1 + l] -= 2.0 * dot * x[l];
            }
        }
    }

    // Extract diagonal and subdiagonal
    for i in 0..n {
        diag[i] = t[i][i];
    }
    for i in 0..n - 1 {
        subdiag[i] = t[i + 1][i];
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Implicit QR for symmetric tridiagonal matrices (Wilkinson shifts)
// ═══════════════════════════════════════════════════════════════════════

/// Implicit QR iteration with Wilkinson shift for a symmetric tridiagonal matrix.
///
/// Operates in-place on `diag` (length n) and `subdiag` (length n-1).
/// Returns (eigenvalues, iterations, converged).
///
/// Reference: Golub & Van Loan, *Matrix Computations*, 4th ed., §8.3.
#[allow(clippy::needless_range_loop)]
fn implicit_qr_tridiag(
    diag: &mut [f64],
    subdiag: &mut [f64],
    tol: f64,
    max_iter: usize,
) -> (Vec<f64>, usize, bool) {
    let n = diag.len();
    if n == 0 { return (vec![], 0, true); }
    if n == 1 { return (vec![diag[0]], 0, true); }

    let mut hi = n - 1;
    let mut total_iters = 0usize;

    // Deflate from the bottom
    while hi > 0 && total_iters < max_iter {
        // Check for negligible subdiagonal elements (deflation)
        for i in 0..hi {
            if subdiag[i].abs() <= tol * (diag[i].abs() + diag[i + 1].abs()) {
                subdiag[i] = 0.0;
            }
        }

        // Find the largest unreduced block at the bottom
        while hi > 0 && subdiag[hi - 1].abs() == 0.0 {
            hi -= 1;
        }
        if hi == 0 { break; }

        // Find the top of the unreduced block
        let mut block_lo = hi - 1;
        while block_lo > 0 && subdiag[block_lo - 1].abs() != 0.0 {
            block_lo -= 1;
        }

        // Wilkinson shift: eigenvalue of the bottom-right 2x2 closer to diag[hi]
        let dd = (diag[hi - 1] - diag[hi]) / 2.0;
        let mu = diag[hi] - subdiag[hi - 1].powi(2)
            / (dd + dd.signum() * (dd * dd + subdiag[hi - 1].powi(2)).sqrt());

        // Implicit QR step (chase the bulge)
        let mut x = diag[block_lo] - mu;
        let mut z = subdiag[block_lo];

        for k in block_lo..hi {
            let (c, s) = givens(x, z);
            let r = x.hypot(z);

            if k > block_lo {
                subdiag[k - 1] = r;
            }

            // Apply Givens rotation G(k, k+1, θ) to rows/cols k, k+1 of T
            let dk = diag[k];
            let dk1 = diag[k + 1];
            let ek = subdiag[k];

            // Updated 2×2 block after similarity transform
            diag[k]     = c * c * dk + 2.0 * c * s * ek + s * s * dk1;
            diag[k + 1] = s * s * dk - 2.0 * c * s * ek + c * c * dk1;
            subdiag[k]  = c * s * (dk1 - dk) + (c * c - s * s) * ek;

            // Prepare for next rotation (chase the bulge)
            if k + 1 < hi {
                x = subdiag[k];
                z = s * subdiag[k + 1];
                subdiag[k + 1] *= c;
            }
        }

        total_iters += 1;
    }

    let converged = hi == 0;
    (diag.to_vec(), total_iters, converged)
}

/// Compute Givens rotation parameters (c, s) such that [c s; -s c]^T [a; b] = [r; 0].
fn givens(a: f64, b: f64) -> (f64, f64) {
    if b.abs() < 1e-300 {
        (1.0, 0.0)
    } else if a.abs() < 1e-300 {
        (0.0, b.signum())
    } else {
        let r = a.hypot(b);
        (a / r, b / r)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Lanczos iteration for sparse symmetric matrices
// ═══════════════════════════════════════════════════════════════════════

/// Lanczos iteration with full reorthogonalization.
///
/// Computes the k smallest eigenvalues of the sparse Laplacian.
/// Returns (eigenvalues, iterations, converged).
#[allow(clippy::needless_range_loop)]
fn lanczos_eigenvalues(
    mat: &SparseCathedralProbe,
    k: usize,
    _max_iter: usize,
    tol: f64,
) -> (Vec<f64>, usize, bool) {
    let n = mat.node_count();
    let m = (2 * k + 10).min(n); // Lanczos subspace dimension

    let mut alpha = vec![0.0f64; m];
    let mut beta = vec![0.0f64; m];
    let mut q = vec![vec![0.0f64; n]; m + 1];

    // Random starting vector (deterministic seed via simple LCG)
    let mut rng_state: u64 = 12345;
    for i in 0..n {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        q[0][i] = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
    }
    // Normalize
    let q0_norm = q[0].iter().map(|&v| v * v).sum::<f64>().sqrt();
    if q0_norm < 1e-15 {
        q[0][0] = 1.0;
    } else {
        for qi in q[0].iter_mut() {
        *qi /= q0_norm;
    }
    }

    let mut w = vec![0.0f64; n];
    let mut iters_used = 0;
    let mut j = 0;

    for iter in 0..m {
        j = iter;
        // w = A * q[j]
        mat.matvec(&q[j], &mut w);

        // alpha[j] = q[j]^T * w
        alpha[j] = (0..n).map(|i| q[j][i] * w[i]).sum();

        // w = w - alpha[j]*q[j] - beta[j-1]*q[j-1]
        for i in 0..n {
            w[i] -= alpha[j] * q[j][i];
            if iter > 0 {
                w[i] -= beta[iter - 1] * q[iter - 1][i];
            }
        }

        // Full reorthogonalization (two passes for numerical stability)
        for _ in 0..2 {
            for l in 0..=iter {
                let dot: f64 = (0..n).map(|i| w[i] * q[l][i]).sum();
                for i in 0..n {
                    w[i] -= dot * q[l][i];
                }
            }
        }

        let w_norm: f64 = w.iter().map(|&v| v * v).sum::<f64>().sqrt();
        beta[iter] = w_norm;

        if w_norm < tol {
            j = iter;
            iters_used = iter + 1;
            break;
        }

        if iter + 1 < m {
            let inv = 1.0 / w_norm;
            for i in 0..n {
                q[iter + 1][i] = w[i] * inv;
            }
        }
        iters_used = iter + 1;
    }

    // Now compute eigenvalues of the (j+1)×(j+1) tridiagonal matrix [alpha, beta]
    let sz = j + 1;
    let mut t_diag = alpha[..sz].to_vec();
    let mut t_sub = beta[..sz.saturating_sub(1)].to_vec();

    let (mut eigs, qr_iters, converged) = implicit_qr_tridiag(&mut t_diag, &mut t_sub, tol * 100.0, sz * 30);

    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Clean near-zero
    for e in &mut eigs {
        if e.abs() < 1e-10 { *e = 0.0; }
    }

    let k_actual = k.min(eigs.len());
    (eigs[..k_actual].to_vec(), iters_used + qr_iters, converged)
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Helper ─────────────────────────────────────────────────

    /// Relative tolerance check.
    fn rel_close(actual: f64, expected: f64, tol: f64) -> bool {
        if expected.abs() < 1e-10 {
            actual.abs() < tol
        } else {
            (actual - expected).abs() / expected.abs() < tol
        }
    }

    /// Build K_n (complete graph on n nodes, weight 1.0).
    fn complete_graph(n: usize) -> CathedralProbe {
        let names: Vec<String> = (0..n).map(|i| format!("n{i}")).collect();
        let mut g = CathedralProbe::new(names.iter().map(|s| s.as_str()).collect());
        for i in 0..n {
            for j in (i + 1)..n {
                g.connect(&names[i], &names[j], 1.0);
            }
        }
        g
    }

    /// Build P_n (path graph on n nodes, weight 1.0).
    fn path_graph(n: usize) -> CathedralProbe {
        let names: Vec<String> = (0..n).map(|i| format!("n{i}")).collect();
        let mut g = CathedralProbe::new(names.iter().map(|s| s.as_str()).collect());
        for i in 0..n - 1 {
            g.connect(&names[i], &names[i + 1], 1.0);
        }
        g
    }

    /// Build S_n (star graph: 1 hub + n-1 leaves, weight 1.0).
    fn star_graph(n: usize) -> CathedralProbe {
        let names: Vec<String> = (0..n).map(|i| format!("n{i}")).collect();
        let mut g = CathedralProbe::new(names.iter().map(|s| s.as_str()).collect());
        for i in 1..n {
            g.connect(&names[0], &names[i], 1.0);
        }
        g
    }

    // ─── Basic construction ─────────────────────────────────────

    #[test]
    fn test_create_probe() {
        let p = CathedralProbe::new(vec!["a", "b", "c"]);
        assert_eq!(p.node_count(), 3);
        assert_eq!(p.edge_count(), 0);
    }

    #[test]
    fn test_add_edges() {
        let mut p = CathedralProbe::new(vec!["a", "b", "c"]);
        p.connect("a", "b", 1.0);
        p.connect("b", "c", 1.0);
        assert_eq!(p.edge_count(), 2);
    }

    #[test]
    fn test_components() {
        let p = CathedralProbe::new(vec!["web", "api", "db"]);
        assert_eq!(p.components(), &["web", "api", "db"]);
    }

    #[test]
    fn test_empty_graph() {
        let p = CathedralProbe::new(vec![]);
        assert!(p.spectrum().is_empty());
        assert_eq!(p.fiedler_value(), 0.0);
        assert!(p.is_connected());
    }

    #[test]
    fn test_single_node() {
        let p = CathedralProbe::new(vec!["solo"]);
        let spec = p.spectrum();
        assert_eq!(spec.len(), 1);
        assert_eq!(spec[0], 0.0);
    }

    // ─── Exact eigenvalue tests (known values) ──────────────────

    #[test]
    fn test_k3_spectrum_exact() {
        // K₃: eigenvalues = {0, 3, 3}
        let p = complete_graph(3);
        let spec = p.spectrum();
        assert_eq!(spec.len(), 3);
        assert!(spec[0].abs() < 0.01, "λ₀ = {} (expected 0)", spec[0]);
        assert!(
            rel_close(spec[1], 3.0, 0.01),
            "λ₁ = {} (expected 3.0)", spec[1]
        );
        assert!(
            rel_close(spec[2], 3.0, 0.01),
            "λ₂ = {} (expected 3.0)", spec[2]
        );
    }

    #[test]
    fn test_k3_fiedler_exact() {
        // K₃: Fiedler = 3.0
        let p = complete_graph(3);
        let f = p.fiedler_value();
        assert!(
            rel_close(f, 3.0, 0.01),
            "K₃ Fiedler = {f} (expected 3.0)"
        );
    }

    #[test]
    fn test_k4_spectrum_exact() {
        // K₄: eigenvalues = {0, 4, 4, 4}
        let p = complete_graph(4);
        let spec = p.spectrum();
        assert_eq!(spec.len(), 4);
        assert!(spec[0].abs() < 0.05);
        assert!(rel_close(spec[1], 4.0, 0.02), "λ₁ = {}", spec[1]);
        assert!(rel_close(spec[2], 4.0, 0.02), "λ₂ = {}", spec[2]);
        assert!(rel_close(spec[3], 4.0, 0.02), "λ₃ = {}", spec[3]);
    }

    #[test]
    fn test_p4_spectrum_exact() {
        // P₄ (path on 4 nodes): eigenvalues = {0, 2-√2, 2, 2+√2}
        // Fiedler = 2 - √2 ≈ 0.5858
        let p = path_graph(4);
        let spec = p.spectrum();
        assert_eq!(spec.len(), 4);
        let fiedler = 2.0 - 2.0_f64.sqrt();
        assert!(
            rel_close(spec[1], fiedler, 0.02),
            "P₄ Fiedler = {} (expected {fiedler:.4})", spec[1]
        );
        assert!(
            rel_close(spec[2], 2.0, 0.02),
            "P₄ λ₂ = {} (expected 2.0)", spec[2]
        );
        let max_eig = 2.0 + 2.0_f64.sqrt();
        assert!(
            rel_close(spec[3], max_eig, 0.02),
            "P₄ λ₃ = {} (expected {max_eig:.4})", spec[3]
        );
    }

    #[test]
    fn test_p4_fiedler_exact() {
        // P₄: Fiedler = 2 - √2 ≈ 0.5858
        let p = path_graph(4);
        let f = p.fiedler_value();
        let expected = 2.0 - 2.0_f64.sqrt();
        assert!(
            rel_close(f, expected, 0.02),
            "P₄ Fiedler = {f:.4} (expected {expected:.4})"
        );
    }

    #[test]
    fn test_s4_fiedler_exact() {
        // S₄ (star: 1 hub + 3 leaves): Fiedler = 1.0
        // Eigenvalues = {0, 1, 1, 4}
        let p = star_graph(4);
        let f = p.fiedler_value();
        assert!(
            rel_close(f, 1.0, 0.02),
            "S₄ Fiedler = {f:.4} (expected 1.0)"
        );
    }

    #[test]
    fn test_s4_spectrum_exact() {
        // S₄: {0, 1, 1, 4}
        let p = star_graph(4);
        let spec = p.spectrum();
        assert_eq!(spec.len(), 4);
        assert!(spec[0].abs() < 0.05);
        assert!(rel_close(spec[1], 1.0, 0.02), "λ₁ = {}", spec[1]);
        assert!(rel_close(spec[2], 1.0, 0.02), "λ₂ = {}", spec[2]);
        assert!(rel_close(spec[3], 4.0, 0.02), "λ₃ = {}", spec[3]);
    }

    #[test]
    fn test_p3_fiedler_exact() {
        // P₃ (path on 3): eigenvalues = {0, 1, 3}, Fiedler = 1.0
        let p = path_graph(3);
        let spec = p.spectrum();
        assert!(spec[0].abs() < 0.05);
        assert!(rel_close(spec[1], 1.0, 0.02));
        assert!(rel_close(spec[2], 3.0, 0.02));
    }

    #[test]
    fn test_k2_spectrum_exact() {
        // K₂ (two nodes, one edge w=1): eigenvalues = {0, 2}
        let mut p = CathedralProbe::new(vec!["a", "b"]);
        p.connect("a", "b", 1.0);
        let spec = p.spectrum();
        assert_eq!(spec.len(), 2);
        assert!(spec[0].abs() < 0.01);
        assert!(rel_close(spec[1], 2.0, 0.01));
    }

    fn test_p5_fiedler_exact() {
        // P₅: Fiedler = 2 - 2cos(π/5) ≈ 0.3820
        let p = path_graph(5);
        let f = p.fiedler_value();
        let expected = 2.0 - 2.0 * (std::f64::consts::PI / 5.0).cos();
        assert!(
            rel_close(f, expected, 0.02),
            "P₅ Fiedler = {f:.4} (expected {expected:.4})"
        );
    }

    #[test]
    fn test_weighted_k2_exact() {
        // K₂ with weight w: eigenvalues = {0, 2w}
        let mut p = CathedralProbe::new(vec!["a", "b"]);
        p.connect("a", "b", 3.5);
        let spec = p.spectrum();
        assert!(spec[0].abs() < 0.01);
        assert!(rel_close(spec[1], 7.0, 0.01));
    }

    #[test]
    fn test_cycle_c4_fiedler_exact() {
        // C₄ (cycle on 4): eigenvalues = {0, 2, 2, 4}, Fiedler = 2.0
        let names = vec!["a", "b", "c", "d"];
        let mut g = CathedralProbe::new(names.clone());
        g.connect("a", "b", 1.0);
        g.connect("b", "c", 1.0);
        g.connect("c", "d", 1.0);
        g.connect("d", "a", 1.0);
        let f = g.fiedler_value();
        assert!(
            rel_close(f, 2.0, 0.02),
            "C₄ Fiedler = {f:.4} (expected 2.0)"
        );
    }

    // ─── Convergence and spectrum_result ────────────────────────

    #[test]
    fn test_spectrum_result_convergence() {
        let p = complete_graph(5);
        let result = p.spectrum_result();
        assert!(result.converged, "QR should converge for small symmetric matrices");
        assert!(result.iterations > 0 || p.node_count() <= 2);
        assert_eq!(result.method, SpectrumMethod::DenseImplicitQr);
        assert_eq!(result.eigenvalues.len(), 5);
    }

    #[test]
    fn test_spectrum_result_empty() {
        let p = CathedralProbe::new(vec![]);
        let result = p.spectrum_result();
        assert!(result.converged);
        assert!(result.eigenvalues.is_empty());
    }

    #[test]
    fn test_spectrum_result_fiedler() {
        let mut p = CathedralProbe::new(vec!["a", "b"]);
        p.connect("a", "b", 1.0);
        let result = p.spectrum_result();
        let f = result.fiedler_value().unwrap();
        assert!(rel_close(f, 2.0, 0.01));
    }

    #[test]
    fn test_spectrum_result_fiedler_empty() {
        let p = CathedralProbe::new(vec![]);
        assert!(p.spectrum_result().fiedler_value().is_none());
    }

    // ─── Connectivity ───────────────────────────────────────────

    #[test]
    fn test_fiedler_connected() {
        let mut p = CathedralProbe::new(vec!["a", "b"]);
        p.connect("a", "b", 1.0);
        assert!(rel_close(p.fiedler_value(), 2.0, 0.01));
    }

    #[test]
    fn test_fiedler_disconnected() {
        let p = CathedralProbe::new(vec!["a", "b"]);
        assert!(p.fiedler_value() < 0.01);
    }

    #[test]
    fn test_is_connected() {
        let mut p = CathedralProbe::new(vec!["a", "b", "c"]);
        p.connect("a", "b", 1.0);
        p.connect("b", "c", 1.0);
        assert!(p.is_connected());
    }

    #[test]
    fn test_is_not_connected() {
        let mut p = CathedralProbe::new(vec!["a", "b", "c"]);
        p.connect("a", "b", 1.0);
        assert!(!p.is_connected());
    }

    #[test]
    fn test_connected_components_disconnected() {
        let mut p = CathedralProbe::new(vec!["a", "b", "c", "d"]);
        p.connect("a", "b", 1.0);
        assert!(p.connected_components() >= 2);
    }

    #[test]
    fn test_connected_components_complete() {
        let p = complete_graph(4);
        assert_eq!(p.connected_components(), 1);
    }

    // ─── Cheeger inequality ─────────────────────────────────────

    #[test]
    fn test_cheeger_bounds() {
        let mut p = CathedralProbe::new(vec!["a", "b"]);
        p.connect("a", "b", 1.0);
        let ub = p.cheeger_upper_bound();
        let lb = p.cheeger_lower_bound();
        assert!(ub >= 0.0);
        assert!(lb >= 0.0);
        assert!(ub >= lb - 1e-10);
    }

    #[test]
    fn test_cheeger_legacy_alias() {
        let mut p = CathedralProbe::new(vec!["a", "b"]);
        p.connect("a", "b", 1.0);
        #[allow(deprecated)]
        let c = p.cheeger_constant();
        assert_eq!(c, p.cheeger_upper_bound());
    }

    #[test]
    fn test_cheeger_k3_exact() {
        // K₃: Fiedler = 3, upper = √6 ≈ 2.449, lower = 1.5
        let p = complete_graph(3);
        let ub = p.cheeger_upper_bound();
        let lb = p.cheeger_lower_bound();
        assert!(rel_close(ub, 6.0_f64.sqrt(), 0.02));
        assert!(rel_close(lb, 1.5, 0.02));
    }

    // ─── Fragility ──────────────────────────────────────────────

    #[test]
    fn test_fragility_connected() {
        let mut p = CathedralProbe::new(vec!["a", "b"]);
        p.connect("a", "b", 1.0);
        assert!(p.fragility_index().is_finite());
        assert!(rel_close(p.fragility_index(), 0.5, 0.01)); // 1/2
    }

    #[test]
    fn test_fragility_disconnected() {
        let p = CathedralProbe::new(vec!["a", "b"]);
        assert!(p.fragility_index().is_infinite());
    }

    // ─── Weights and degrees ────────────────────────────────────

    #[test]
    fn test_total_weight() {
        let mut p = CathedralProbe::new(vec!["a", "b", "c"]);
        p.connect("a", "b", 2.0);
        p.connect("b", "c", 3.0);
        assert!((p.total_weight() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_average_degree() {
        let p = complete_graph(3);
        // total_weight = 3, avg_degree = 3*2/3 = 2.0
        assert!((p.average_degree() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_weighted_edges() {
        let mut p = CathedralProbe::new(vec!["a", "b", "c"]);
        p.connect("a", "b", 10.0);
        p.connect("b", "c", 0.1);
        let f = p.fiedler_value();
        assert!(f > 0.0);
    }

    #[test]
    fn test_two_nodes_zero_weight() {
        let mut p = CathedralProbe::new(vec!["a", "b"]);
        p.connect("a", "b", 0.0);
        assert!(p.fiedler_value() < 0.01);
    }

    // ─── Component importance and bottlenecks ───────────────────

    #[test]
    fn test_component_importance() {
        let mut p = CathedralProbe::new(vec!["hub", "s1", "s2"]);
        p.connect("hub", "s1", 1.0);
        p.connect("hub", "s2", 1.0);
        let imp = p.component_importance();
        assert!(imp.contains_key("hub"));
        assert!(imp.contains_key("s1"));
        assert!(imp["hub"] >= imp["s1"] - 1e-10);
    }

    #[test]
    fn test_bottlenecks() {
        let mut p = CathedralProbe::new(vec!["a", "b", "c"]);
        p.connect("a", "b", 1.0);
        p.connect("b", "c", 1.0);
        let bn = p.bottlenecks();
        assert_eq!(bn.len(), 2);
    }

    #[test]
    fn test_is_healthy() {
        let p = complete_graph(3);
        assert!(p.is_healthy(0.1));
    }

    #[test]
    fn test_algebraic_connectivity_alias() {
        let mut p = CathedralProbe::new(vec!["a", "b"]);
        p.connect("a", "b", 1.0);
        assert!((p.algebraic_connectivity() - p.fiedler_value()).abs() < 1e-10);
    }

    // ─── Large graph ────────────────────────────────────────────

    #[test]
    fn test_line_topology_20_nodes() {
        let p = path_graph(20);
        assert_eq!(p.node_count(), 20);
        assert_eq!(p.edge_count(), 19);
        assert!(p.is_connected());
        assert!(p.fiedler_value() > 0.0);
    }

    #[test]
    fn test_star_topology() {
        let p = star_graph(5);
        assert!(p.is_connected());
        assert!(p.fiedler_value() > 0.0);
    }

    #[test]
    fn test_line_vs_complete_fiedler() {
        let line = path_graph(5);
        let complete = complete_graph(5);
        // Complete graph should have higher Fiedler than path
        assert!(
            complete.fiedler_value() > line.fiedler_value(),
            "K₅ Fiedler ({}) should exceed P₅ Fiedler ({})",
            complete.fiedler_value(),
            line.fiedler_value()
        );
    }

    // ─── Sparse graph tests ─────────────────────────────────────

    #[test]
    fn test_sparse_from_edges() {
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)];
        let sparse = SparseCathedralProbe::from_edges(3, &edges);
        assert_eq!(sparse.node_count(), 3);
    }

    #[test]
    fn test_sparse_k3_fiedler() {
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)];
        let sparse = SparseCathedralProbe::from_edges(3, &edges);
        let f = sparse.fiedler_value();
        assert!(
            rel_close(f, 3.0, 0.05),
            "Sparse K₃ Fiedler = {f:.4} (expected 3.0)"
        );
    }

    #[test]
    fn test_sparse_p4_fiedler() {
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)];
        let sparse = SparseCathedralProbe::from_edges(4, &edges);
        let f = sparse.fiedler_value();
        let expected = 2.0 - 2.0_f64.sqrt();
        assert!(
            rel_close(f, expected, 0.05),
            "Sparse P₄ Fiedler = {f:.4} (expected {expected:.4})"
        );
    }

    #[test]
    fn test_sparse_vs_dense_agreement() {
        let mut dense = CathedralProbe::new(vec!["a", "b", "c", "d"]);
        dense.connect("a", "b", 1.0);
        dense.connect("b", "c", 2.0);
        dense.connect("c", "d", 1.0);
        dense.connect("a", "d", 3.0);

        let sparse = SparseCathedralProbe::from(&dense);

        let dense_f = dense.fiedler_value();
        let sparse_f = sparse.fiedler_value();
        assert!(
            (dense_f - sparse_f).abs() / dense_f.abs().max(1e-10) < 0.05,
            "Dense Fiedler = {dense_f:.4}, Sparse = {sparse_f:.4}"
        );
    }

    #[test]
    fn test_sparse_spectrum_top_k() {
        let edges: Vec<(usize, usize, f64)> = (0..9)
            .map(|i| (i, i + 1, 1.0))
            .collect();
        let sparse = SparseCathedralProbe::from_edges(10, &edges);
        let result = sparse.spectrum_top_k(3).unwrap();
        assert_eq!(result.eigenvalues.len(), 3);
        assert!(result.eigenvalues[0].abs() < 0.1); // ~0
        assert!(result.eigenvalues[1] > 0.0); // Fiedler > 0
        assert!(result.converged);
    }

    #[test]
    fn test_sparse_total_weight() {
        let edges = vec![(0, 1, 2.0), (1, 2, 3.0)];
        let sparse = SparseCathedralProbe::from_edges(3, &edges);
        assert!((sparse.total_weight() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_sparse_with_names() {
        let edges = vec![(0, 1, 1.0)];
        let sparse = SparseCathedralProbe::from_edges(2, &edges)
            .with_names(vec!["alice".into(), "bob".into()]);
        assert_eq!(sparse.node_count(), 2);
    }

    // ─── Directed graph tests ───────────────────────────────────

    #[test]
    fn test_directed_cycle_3_fiedler() {
        // Strongly connected directed cycle on 3 nodes
        let mut dg = DirectedCathedralProbe::new(vec!["a", "b", "c"]);
        dg.add_edge("a", "b", 1.0);
        dg.add_edge("b", "c", 1.0);
        dg.add_edge("c", "a", 1.0);
        let f = dg.fiedler_value();
        assert!(f > 0.0, "Strongly connected directed cycle should have Fiedler > 0, got {f}");
    }

    #[test]
    fn test_directed_disconnected_fiedler() {
        // Not strongly connected: a → b, a → c (can't reach a from b or c)
        let mut dg = DirectedCathedralProbe::new(vec!["a", "b", "c"]);
        dg.add_edge("a", "b", 1.0);
        dg.add_edge("a", "c", 1.0);
        // Teleportation gives small positive Fiedler, but it should be much smaller
        // than a strongly connected graph of the same size
        let f = dg.fiedler_value();
        let mut dg2 = DirectedCathedralProbe::new(vec!["a", "b", "c"]);
        dg2.add_edge("a", "b", 1.0);
        dg2.add_edge("b", "c", 1.0);
        dg2.add_edge("c", "a", 1.0);
        let f2 = dg2.fiedler_value();
        assert!(
            f2 > f,
            "Strongly connected Fiedler ({f2}) should exceed weakly connected ({f})"
        );
    }

    #[test]
    fn test_directed_spectrum_result() {
        let mut dg = DirectedCathedralProbe::new(vec!["x", "y"]);
        dg.add_edge("x", "y", 1.0);
        dg.add_edge("y", "x", 1.0);
        let result = dg.spectrum_result();
        assert_eq!(result.eigenvalues.len(), 2);
        assert!(result.eigenvalues[0].abs() < 0.1); // ~0
    }

    #[test]
    fn test_directed_empty() {
        let mut dg = DirectedCathedralProbe::new(vec![]);
        let result = dg.spectrum_result();
        assert!(result.eigenvalues.is_empty());
        assert_eq!(dg.node_count(), 0);
    }

    #[test]
    fn test_directed_single_node() {
        let mut dg = DirectedCathedralProbe::new(vec!["only"]);
        let result = dg.spectrum_result();
        assert_eq!(result.eigenvalues.len(), 1);
    }

    // ─── Error types ────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let e = CathedralError::NodeNotFound("foo".into());
        assert!(e.to_string().contains("foo"));

        let e = CathedralError::InsufficientNodes { have: 1, need: 2 };
        assert!(e.to_string().contains("1"));

        let e = CathedralError::EmptyGraph;
        assert!(e.to_string().contains("empty"));
    }

    // ─── Givens rotation ────────────────────────────────────────

    #[test]
    fn test_givens_identity() {
        let (c, s) = givens(1.0, 0.0);
        assert!((c - 1.0).abs() < 1e-10);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_givens_rotation() {
        let (c, s) = givens(3.0, 4.0);
        // Should zero out the second element
        let r = c * 3.0 + s * 4.0;
        let zero = -s * 3.0 + c * 4.0;
        assert!(r > 0.0);
        assert!(zero.abs() < 1e-10);
        assert!((c * c + s * s - 1.0).abs() < 1e-10);
    }

    // ─── Householder tridiagonalization ─────────────────────────

    #[test]
    fn test_tridiag_preserves_trace() {
        // Trace of A = sum of eigenvalues = sum of diagonal of tridiagonal
        let mut p = CathedralProbe::new(vec!["a", "b", "c", "d"]);
        p.connect("a", "b", 1.0);
        p.connect("b", "c", 2.0);
        p.connect("c", "d", 1.0);
        p.connect("a", "d", 3.0);

        let lap = p.build_laplacian();
        let trace_orig: f64 = (0..4).map(|i| lap[i][i]).sum();

        let mut diag = vec![0.0; 4];
        let mut subdiag = vec![0.0; 3];
        householder_tridiag(&lap, &mut diag, &mut subdiag);

        let trace_tri: f64 = diag.iter().sum();
        assert!(
            (trace_orig - trace_tri).abs() < 1e-8,
            "Trace not preserved: orig={trace_orig}, tri={trace_tri}"
        );
    }

    #[test]
    fn test_tridiag_preserves_frobenius() {
        // For symmetric: ||A||_F = sum of eigenvalues squared = ||T||_F
        let mut p = CathedralProbe::new(vec!["a", "b", "c"]);
        p.connect("a", "b", 1.0);
        p.connect("b", "c", 1.0);
        p.connect("a", "c", 1.0);

        let lap = p.build_laplacian();
        let frob_orig: f64 = lap.iter()
            .flat_map(|row| row.iter())
            .map(|&v| v * v)
            .sum();

        let mut diag = vec![0.0; 3];
        let mut subdiag = vec![0.0; 2];
        householder_tridiag(&lap, &mut diag, &mut subdiag);

        let frob_tri: f64 = diag.iter().map(|&d| d * d).sum::<f64>()
            + 2.0 * subdiag.iter().map(|&s| s * s).sum::<f64>();
        assert!(
            (frob_orig - frob_tri).abs() / frob_orig < 1e-6,
            "Frobenius not preserved: orig={frob_orig}, tri={frob_tri}"
        );
    }
}
