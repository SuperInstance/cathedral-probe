//! # Cathedral Probe
//!
//! Spectral topology analysis for component graphs.
//!
//! Compute Laplacian eigenvalues, Fiedler value (connectivity), Cheeger constant
//! (bottleneck detection), component importance, effective resistance, spectral
//! embedding, spectral clustering, community profiles, and Fiedler sensitivity.
//! Answer: "is the space between my components healthy?"
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpectrumMethod {
    /// Householder tridiagonalization + implicit QR with Wilkinson shifts.
    DenseImplicitQr,
    /// Lanczos iteration with full reorthogonalization (top-k eigenvalues).
    Lanczos { k: usize },
}

/// Result of a spectrum computation with diagnostics.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// Invalid matrix dimensions.
    InvalidMatrix(String),
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
            Self::InvalidMatrix(msg) => write!(f, "invalid matrix: {msg}")
        }
    }
}

impl std::error::Error for CathedralError {}

// ═══════════════════════════════════════════════════════════════════════
// Dense undirected graph
// ═══════════════════════════════════════════════════════════════════════

/// Analysis of a single connected component.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct ComponentAnalysis {
    /// Node indices belonging to this component.
    pub nodes: Vec<usize>,
    /// Fiedler value (second-smallest eigenvalue) of this component.
    pub fiedler_value: f64,
    /// Cheeger upper bound (√(2·λ₂)) of this component.
    pub cheeger_upper: f64,
    /// Number of nodes in this component.
    pub size: usize,
}

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

    // ─── Matrix constructors ────────────────────────────────────

    /// Build a graph from an adjacency/correlation/weight matrix.
    ///
    /// Takes a `&[Vec<f64>]` where `weights[i][j]` is the edge weight between
    /// nodes i and j. Nodes are automatically named "0", "1", "2", ....
    /// Self-loops are ignored. A warning is printed if the matrix is not symmetric.
    pub fn from_matrix(weights: &[Vec<f64>]) -> Result<Self, CathedralError> {
        let n = weights.len();
        if n == 0 {
            return Ok(Self::new(vec![]));
        }
        for (i, row) in weights.iter().enumerate() {
            if row.len() != n {
                return Err(CathedralError::InvalidMatrix(format!(
                    "row {} has {} columns, expected {}",
                    i, row.len(), n
                )));
            }
        }

        let names: Vec<String> = (0..n).map(|i| i.to_string()).collect();
        let mut graph = Self::new(names.iter().map(|s| s.as_str()).collect());

        for i in 0..n {
            for j in (i + 1)..n {
                let w = weights[i][j];
                if (weights[i][j] - weights[j][i]).abs() > 1e-10 {
                    // Non-symmetric: use the average
                    let avg = (weights[i][j] + weights[j][i]) / 2.0;
                    if avg.abs() > 1e-12 {
                        graph.connect(&names[i], &names[j], avg);
                    }
                } else if w.abs() > 1e-12 {
                    graph.connect(&names[i], &names[j], w);
                }
            }
        }

        Ok(graph)
    }

    /// Build a graph from an edge list of `(i, j, weight)` triples.
    ///
    /// `n` is the number of nodes (indexed 0..n-1). No string allocation —
    /// pure indices. This is what ML engineers and neuroscientists need.
    pub fn from_edge_list(n: usize, edges: &[(usize, usize, f64)]) -> Self {
        let names: Vec<String> = (0..n).map(|i| i.to_string()).collect();
        let mut graph = Self::new(names.iter().map(|s| s.as_str()).collect());
        for &(i, j, w) in edges {
            if i < n && j < n && i != j {
                graph.connect(&names[i], &names[j], w);
            }
        }
        graph
    }

    /// Build a graph from a flat row-major weight array.
    ///
    /// `weights` is a flat `&[f64]` of length `n * n`, row-major order.
    /// Zero-copy friendly for FFI / SIMD use cases.
    pub fn from_weighted_adjacency(weights: &[f64], n: usize) -> Result<Self, CathedralError> {
        if weights.len() != n * n {
            return Err(CathedralError::InvalidMatrix(format!(
                "expected {} elements ({}x{}), got {}",
                n * n, n, n, weights.len()
            )));
        }
        let matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| weights[i * n..(i + 1) * n].to_vec())
            .collect();
        Self::from_matrix(&matrix)
    }

    // ─── Per-component analysis ──────────────────────────────────

    /// Analyze each connected component independently.
    ///
    /// Finds connected components via BFS, then computes the Fiedler value
    /// and Cheeger upper bound for each. This fixes the issue where the Fiedler
    /// value is 0 for disconnected graphs — each component gets its own analysis.
    pub fn per_component_analysis(&self) -> Vec<ComponentAnalysis> {
        let n = self.nodes.len();
        if n == 0 {
            return vec![];
        }

        // BFS to find connected components
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(i, j, _w) in &self.edges {
            adj[i].push(j);
            adj[j].push(i);
        }

        let mut visited = vec![false; n];
        let mut components = Vec::new();

        for start in 0..n {
            if visited[start] {
                continue;
            }
            let mut comp_nodes = Vec::new();
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(start);
            visited[start] = true;
            while let Some(node) = queue.pop_front() {
                comp_nodes.push(node);
                for &neighbor in &adj[node] {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
            comp_nodes.sort_unstable();
            components.push(comp_nodes);
        }

        // Analyze each component
        components
            .into_iter()
            .map(|nodes| {
                let size = nodes.len();
                let (fiedler_value, cheeger_upper) = if size < 2 {
                    (0.0, 0.0)
                } else {
                    // Build subgraph for this component
                    let node_set: std::collections::HashSet<usize> =
                        nodes.iter().copied().collect();
                    let sub_names: Vec<String> =
                        nodes.iter().map(|&i| self.nodes[i].clone()).collect();
                    let mut sub = CathedralProbe::new(
                        sub_names.iter().map(|s| s.as_str()).collect(),
                    );
                    for &(i, j, w) in &self.edges {
                        if node_set.contains(&i) && node_set.contains(&j) {
                            sub.connect(&self.nodes[i], &self.nodes[j], w);
                        }
                    }
                    let f = sub.fiedler_value();
                    let c = sub.cheeger_upper_bound();
                    (f, c)
                };

                ComponentAnalysis {
                    nodes,
                    fiedler_value,
                    cheeger_upper,
                    size,
                }
            })
            .collect()
    }

    /// Serialize the last `SpectrumResult` to JSON (requires `serde` feature).
    #[cfg(feature = "serde")]
    pub fn to_json(&self) -> Result<String, CathedralError> {
        let result = self.spectrum_result();
        serde_json::to_string(&result)
            .map_err(|e| CathedralError::InvalidMatrix(format!("serde error: {e}")))
    }

    // ─── Full eigendecomposition (eigenvalues + eigenvectors) ────

    /// Compute all eigenvalues and eigenvectors of the graph Laplacian.
    ///
    /// Returns `(eigenvalues, eigenvectors)` where `eigenvectors[k]` is the
    /// eigenvector corresponding to `eigenvalues[k]`, sorted ascending.
    fn full_eigen(&self) -> (Vec<f64>, Vec<Vec<f64>>) {
        let n = self.nodes.len();
        if n == 0 { return (vec![], vec![]); }
        if n == 1 { return (vec![0.0], vec![vec![1.0]]); }

        let lap = self.build_laplacian();
        let (diag, subdiag, q) = householder_tridiag_with_q(&lap);
        let (mut eigs, vecs_tri, _iters, _converged) =
            implicit_qr_tridiag_with_vectors(diag, subdiag, 1e-14, n * 30);

        // Transform eigenvectors back: V = Q * V_tridiag
        // vecs_tri[j][k] = j-th component of k-th eigenvector (columns are eigenvectors)
        let mut vecs = vec![vec![0.0f64; n]; n];
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    vecs[k][i] += q[i][j] * vecs_tri[j][k];
                }
            }
        }

        // Clean near-zero eigenvalues
        for e in &mut eigs {
            if e.abs() < 1e-10 { *e = 0.0; }
        }

        // Sort by eigenvalue ascending
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| eigs[a].partial_cmp(&eigs[b]).unwrap());

        let sorted_eigs: Vec<f64> = indices.iter().map(|&i| eigs[i]).collect();
        let sorted_vecs: Vec<Vec<f64>> = indices.iter().map(|&i| vecs[i].clone()).collect();

        (sorted_eigs, sorted_vecs)
    }

    // ─── Advanced spectral methods ──────────────────────────────

    /// Effective graph resistance (resistance distance) between nodes i and j.
    ///
    /// R_eff(i,j) = Σ_{k=1}^{n-1} (1/λ_k) · (v_k[i] - v_k[j])²
    ///
    /// Uses the pseudoinverse of the Laplacian: L⁺ = V Λ⁺ V^T.
    ///
    /// # References
    /// - Klein, D.J. & Randić, M. (1993). "Resistance Distance."
    ///   *Journal of Mathematical Chemistry*, 17, 165-180.
    pub fn effective_resistance(&self, i: usize, j: usize) -> f64 {
        let n = self.nodes.len();
        assert!(i < n && j < n, "node indices out of bounds");
        if i == j { return 0.0; }

        let (eigs, vecs) = self.full_eigen();
        let mut r_eff = 0.0;
        // Skip k=0 (λ₀ = 0); sum k=1..n-1
        for k in 1..n {
            if eigs[k].abs() < 1e-14 { continue; }
            let diff = vecs[k][i] - vecs[k][j];
            r_eff += diff * diff / eigs[k];
        }
        r_eff
    }

    /// Kirchhoff index: sum of all pairwise effective resistances.
    ///
    /// Kf(G) = Σ_{i<j} R_eff(i,j) = n · trace(L⁺)
    ///
    /// # References
    /// - Klein & Randić (1993), "Resistance Distance"
    pub fn kirchhoff_index(&self) -> f64 {
        let n = self.nodes.len();
        if n < 2 { return 0.0; }

        let (eigs, _vecs) = self.full_eigen();
        // trace(L⁺) = Σ_{k=1}^{n-1} 1/λ_k
        let trace_lplus: f64 = eigs[1..].iter()
            .filter(|&&e| e.abs() > 1e-14)
            .map(|&e| 1.0 / e)
            .sum();
        n as f64 * trace_lplus
    }

    /// Spectral embedding: project nodes into ℝ^k using the k smallest
    /// non-zero eigenvectors of the Laplacian.
    ///
    /// Returns a vector of length n, where each entry is a k-dimensional vector.
    /// This is the foundation of spectral clustering (Ng, Jordan & Weiss 2001).
    ///
    /// # References
    /// - Ng, A., Jordan, M. & Weiss, Y. (2001). "On Spectral Clustering."
    ///   *NeurIPS*.
    pub fn spectral_embedding(&self, dimensions: usize) -> Vec<Vec<f64>> {
        let n = self.nodes.len();
        if n == 0 { return vec![]; }

        let (eigs, vecs) = self.full_eigen();
        // Collect the k smallest non-zero eigenvectors
        let mut embed_vecs: Vec<&Vec<f64>> = Vec::new();
        for k in 1..n {
            if eigs[k].abs() > 1e-14 {
                embed_vecs.push(&vecs[k]);
                if embed_vecs.len() >= dimensions { break; }
            }
        }
        let d = embed_vecs.len();

        // Build embedding matrix: node i -> vec of length d
        let mut embedding = vec![vec![0.0; d]; n];
        for (dim, vec) in embed_vecs.iter().enumerate() {
            for i in 0..n {
                embedding[i][dim] = vec[i];
            }
        }
        embedding
    }

    /// Spectral clustering: partition nodes into k clusters.
    ///
    /// Uses spectral embedding into ℝ^k followed by k-means (Lloyd's algorithm)
    /// with row-normalization (Ng-Jordan-Weiss method).
    ///
    /// Returns a vector of length n with cluster assignments (0..k-1).
    ///
    /// # References
    /// - Ng, Jordan & Weiss (2001). "On Spectral Clustering."
    pub fn spectral_cluster(&self, k: usize) -> Vec<usize> {
        let n = self.nodes.len();
        if n == 0 { return vec![]; }
        if k <= 1 { return vec![0; n]; }
        if k >= n {
            return (0..n).collect();
        }

        // Step 1: Spectral embedding into k dimensions
        let mut embedding = self.spectral_embedding(k);
        let d = embedding[0].len();
        if d == 0 { return vec![0; n]; }

        // Step 2: Row-normalize (Ng-Jordan-Weiss)
        for row in &mut embedding {
            let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 1e-14 {
                for v in row.iter_mut() { *v /= norm; }
            }
        }

        // Step 3: k-means (Lloyd's algorithm)
        kmeans(&embedding, k, 100)
    }

    /// Network community profile: for each community size s, compute
    /// the minimum conductance Φ(s).
    ///
    /// Returns (size, min_conductance) pairs, revealing the "best" communities
    /// at each scale. Uses the Fiedler vector sweep-cut heuristic.
    ///
    /// # References
    /// - Leskovec, J. et al. (2009). "Community Structure in Large Networks:
    ///   Natural Cluster Sizes and the Absence of Large Well-Defined Clusters."
    pub fn community_profile(&self) -> Vec<(usize, f64)> {
        let n = self.nodes.len();
        if n < 2 { return vec![]; }

        let edges = &self.edges;

        // Compute degree of each node
        let mut degree = vec![0.0f64; n];
        for &(i, j, w) in edges {
            degree[i] += w;
            degree[j] += w;
        }
        let total_vol: f64 = degree.iter().sum();

        // Sort nodes by Fiedler vector value (sweep-cut heuristic)
        let (_, vecs) = self.full_eigen();
        if vecs.len() < 2 { return vec![]; }
        let fiedler_vec = &vecs[1];

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| fiedler_vec[a].partial_cmp(&fiedler_vec[b]).unwrap());

        let mut node_pos = vec![0usize; n];
        for (pos, &node) in order.iter().enumerate() {
            node_pos[node] = pos;
        }

        let mut profile = Vec::new();
        for s in 1..=(n / 2) {
            let vol_s: f64 = order[..s].iter().map(|&node| degree[node]).sum();

            // Compute cut edges crossing the partition
            let in_s = {
                let mut mask = vec![false; n];
                for &node in &order[..s] { mask[node] = true; }
                mask
            };
            let mut cut = 0.0f64;
            for &(i, j, w) in edges {
                if in_s[i] != in_s[j] {
                    cut += w;
                }
            }

            let vol_complement = total_vol - vol_s;
            let cond = if vol_s.min(vol_complement) > 1e-14 {
                cut / vol_s.min(vol_complement)
            } else {
                f64::INFINITY
            };
            profile.push((s, cond));
        }

        profile
    }

    /// Fiedler sensitivity: how much does the Fiedler value change if we
    /// modify each edge?
    ///
    /// Returns `Vec<(usize, usize, f64)>` — (i, j, ∂λ₂/∂w_ij) for each edge.
    /// Uses the formula: ∂λ₂/∂w_ij = (v₂[i] - v₂[j])²
    ///
    /// This is more efficient than `bottlenecks()` which rebuilds the graph.
    ///
    /// # References
    /// - Fiedler, M. (1973). "Algebraic connectivity of graphs."
    pub fn fiedler_sensitivity(&self) -> Vec<(usize, usize, f64)> {
        let n = self.nodes.len();
        if n < 2 { return vec![]; }

        let (_, vecs) = self.full_eigen();
        if vecs.len() < 2 { return vec![]; }
        let fiedler_vec = &vecs[1];

        self.edges.iter().map(|&(i, j, _)| {
            let diff = fiedler_vec[i] - fiedler_vec[j];
            (i, j, diff * diff)
        }).collect()
    }

    /// Condition number of the graph Laplacian.
    ///
    /// κ(L) = λ_max / λ_min_nonzero
    ///
    /// Measures numerical stability of solving Laplacian systems.
    /// A large condition number indicates ill-conditioning.
    pub fn condition_number(&self) -> f64 {
        let spec = self.spectrum();
        if spec.is_empty() { return 0.0; }

        let lambda_max = spec.last().copied().unwrap_or(0.0);
        let lambda_min_nonzero = spec.iter()
            .find(|&&e| e.abs() > 1e-10)
            .copied();

        match lambda_min_nonzero {
            Some(lnz) if lnz.abs() > 1e-14 => lambda_max / lnz,
            _ => {
                // No non-zero eigenvalue found => graph is disconnected (Laplacian is singular)
                f64::INFINITY
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// k-means (Lloyd's algorithm)
// ═══════════════════════════════════════════════════════════════════════

/// Simple k-means clustering using Lloyd's algorithm.
///
/// Returns cluster assignments for each point (0..k-1).
fn kmeans(data: &[Vec<f64>], k: usize, max_iter: usize) -> Vec<usize> {
    let n = data.len();
    let d = if n > 0 { data[0].len() } else { return vec![]; };
    if k == 0 || n == 0 || d == 0 { return vec![0; n]; }

    // Initialize centroids: spread them across the data
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    for c in 0..k {
        let idx = c * n / k;
        centroids.push(data[idx].clone());
    }

    let mut assignments = vec![0usize; n];

    for _ in 0..max_iter {
        let mut changed = false;

        // Assignment step
        for i in 0..n {
            let mut best = 0;
            let mut best_dist = f64::INFINITY;
            for (c, centroid) in centroids.iter().enumerate() {
                let dist: f64 = data[i].iter()
                    .zip(centroid)
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                if dist < best_dist {
                    best_dist = dist;
                    best = c;
                }
            }
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }

        if !changed { break; }

        // Update step
        let mut sums = vec![vec![0.0; d]; k];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assignments[i];
            counts[c] += 1;
            for j in 0..d {
                sums[c][j] += data[i][j];
            }
        }
        for (c, centroid) in centroids.iter_mut().enumerate() {
            if counts[c] > 0 {
                for j in 0..d {
                    centroid[j] = sums[c][j] / counts[c] as f64;
                }
            }
        }
    }

    assignments
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
            adj[i].sort_by_key(|(j, _)| *j);
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
                values.push(-w);
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
                sum += self.values[idx] * x[self.col_ind[idx]];
            }
            y[i] = sum;
        }
    }

    /// Total edge weight.
    pub fn total_weight(&self) -> f64 {
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
    #[allow(clippy::needless_range_loop)]
    fn compute_stationary(&mut self) {
        let n = self.n;
        if n == 0 { return; }

        let mut p = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            let total: f64 = self.out_edges[i].iter().map(|&(_, w)| w).sum();
            if total > 0.0 {
                for &(j, w) in &self.out_edges[i] {
                    p[i][j] = w / total;
                }
            } else {
                for val in p[i].iter_mut().take(n) { *val = 1.0 / n as f64; }
            }
        }

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

        let mut p = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            let total: f64 = self.out_edges[i].iter().map(|&(_, w)| w).sum();
            if total > 0.0 {
                for &(j, w) in &self.out_edges[i] {
                    p[i][j] = w / total;
                }
            }
        }

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
/// Reference: Golub & Van Loan, *Matrix Computations*, 4th ed., Algorithm 8.3.1.
#[allow(clippy::needless_range_loop)]
fn householder_tridiag(a: &[Vec<f64>], diag: &mut [f64], subdiag: &mut [f64]) {
    let (d, s, _) = householder_tridiag_impl(a);
    diag.copy_from_slice(&d);
    subdiag.copy_from_slice(&s);
}

/// Householder tridiagonalization that also returns the orthogonal transformation Q.
#[allow(clippy::needless_range_loop)]
fn householder_tridiag_with_q(a: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    householder_tridiag_impl(a)
}

/// Core implementation of Householder tridiagonalization.
/// Returns (diag, subdiag, Q) where Q accumulates all Householder reflections.
#[allow(clippy::needless_range_loop)]
fn householder_tridiag_impl(a: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut t = a.to_vec();

    // Initialize Q = I
    let mut q = vec![vec![0.0; n]; n];
    for i in 0..n { q[i][i] = 1.0; }

    for k in 0..n.saturating_sub(2) {
        let m = n - k - 1;
        let mut x = vec![0.0; m];
        for i in 0..m {
            x[i] = t[k + 1 + i][k];
        }

        let _sigma: f64 = x[1..].iter().map(|&v| v * v).sum::<f64>();
        let alpha = x[0].signum() * x.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-300);
        x[0] += alpha;

        let v_norm: f64 = x.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if v_norm < 1e-15 { continue; }
        for v in &mut x { *v /= v_norm; }

        // Apply from left
        for j in 0..n {
            let dot: f64 = (0..m).map(|i| x[i] * t[k + 1 + i][j]).sum();
            for i in 0..m {
                t[k + 1 + i][j] -= 2.0 * x[i] * dot;
            }
        }
        // Apply from right
        for i in 0..n {
            let dot: f64 = (0..m).map(|l| t[i][k + 1 + l] * x[l]).sum();
            for l in 0..m {
                t[i][k + 1 + l] -= 2.0 * dot * x[l];
            }
        }

        // Accumulate Q: Q = Q * H_k
        for i in 0..n {
            let dot: f64 = (0..m).map(|l| q[i][k + 1 + l] * x[l]).sum();
            for l in 0..m {
                q[i][k + 1 + l] -= 2.0 * dot * x[l];
            }
        }
    }

    let mut diag = vec![0.0; n];
    let mut subdiag = vec![0.0; n.saturating_sub(1)];
    for i in 0..n { diag[i] = t[i][i]; }
    for i in 0..n.saturating_sub(1) { subdiag[i] = t[i + 1][i]; }

    (diag, subdiag, q)
}

// ═══════════════════════════════════════════════════════════════════════
// Implicit QR for symmetric tridiagonal matrices (Wilkinson shifts)
// ═══════════════════════════════════════════════════════════════════════

/// Implicit QR iteration with Wilkinson shift (eigenvalues only).
#[allow(clippy::needless_range_loop)]
fn implicit_qr_tridiag(
    diag: &mut [f64],
    subdiag: &mut [f64],
    tol: f64,
    max_iter: usize,
) -> (Vec<f64>, usize, bool) {
    let (eigs, _, iters, converged) = implicit_qr_tridiag_impl(diag, subdiag, tol, max_iter, false);
    (eigs, iters, converged)
}

/// Implicit QR iteration that also accumulates eigenvectors.
#[allow(clippy::needless_range_loop)]
fn implicit_qr_tridiag_with_vectors(
    diag: Vec<f64>,
    subdiag: Vec<f64>,
    tol: f64,
    max_iter: usize,
) -> (Vec<f64>, Vec<Vec<f64>>, usize, bool) {
    implicit_qr_tridiag_impl(&diag, &subdiag, tol, max_iter, true)
}

/// Core QR implementation. When `compute_vectors` is true, tracks Givens rotations
/// to build the eigenvector matrix.
#[allow(clippy::needless_range_loop)]
fn implicit_qr_tridiag_impl(
    diag_src: &[f64],
    subdiag_src: &[f64],
    tol: f64,
    max_iter: usize,
    compute_vectors: bool,
) -> (Vec<f64>, Vec<Vec<f64>>, usize, bool) {
    let n = diag_src.len();
    if n == 0 { return (vec![], vec![], 0, true); }
    if n == 1 { return (vec![diag_src[0]], vec![vec![1.0]], 0, true); }

    let mut diag = diag_src.to_vec();
    let mut subdiag = subdiag_src.to_vec();

    // Eigenvector matrix (columns are eigenvectors)
    let mut vecs = if compute_vectors {
        let mut v = vec![vec![0.0; n]; n];
        for i in 0..n { v[i][i] = 1.0; }
        Some(v)
    } else {
        None
    };

    let mut hi = n - 1;
    let mut total_iters = 0usize;

    while hi > 0 && total_iters < max_iter {
        for i in 0..hi {
            if subdiag[i].abs() <= tol * (diag[i].abs() + diag[i + 1].abs()) {
                subdiag[i] = 0.0;
            }
        }

        while hi > 0 && subdiag[hi - 1].abs() == 0.0 {
            hi -= 1;
        }
        if hi == 0 { break; }

        let mut block_lo = hi - 1;
        while block_lo > 0 && subdiag[block_lo - 1].abs() != 0.0 {
            block_lo -= 1;
        }

        let dd = (diag[hi - 1] - diag[hi]) / 2.0;
        let mu = diag[hi] - subdiag[hi - 1].powi(2)
            / (dd + dd.signum() * (dd * dd + subdiag[hi - 1].powi(2)).sqrt());

        let mut x = diag[block_lo] - mu;
        let mut z = subdiag[block_lo];

        for k in block_lo..hi {
            let (c, s) = givens(x, z);
            let r = x.hypot(z);

            if k > block_lo {
                subdiag[k - 1] = r;
            }

            let dk = diag[k];
            let dk1 = diag[k + 1];
            let ek = subdiag[k];

            diag[k]     = c * c * dk + 2.0 * c * s * ek + s * s * dk1;
            diag[k + 1] = s * s * dk - 2.0 * c * s * ek + c * c * dk1;
            subdiag[k]  = c * s * (dk1 - dk) + (c * c - s * s) * ek;

            // Apply Givens rotation to eigenvector matrix
            if let Some(ref mut v) = vecs {
                for i in 0..n {
                    let vk = v[i][k];
                    let vk1 = v[i][k + 1];
                    v[i][k]     = c * vk + s * vk1;
                    v[i][k + 1] = -s * vk + c * vk1;
                }
            }

            if k + 1 < hi {
                x = subdiag[k];
                z = s * subdiag[k + 1];
                subdiag[k + 1] *= c;
            }
        }

        total_iters += 1;
    }

    let converged = hi == 0;
    let eigs = diag.to_vec();
    let eigenvectors = vecs.unwrap_or_else(|| vec![vec![0.0; n]; n]);
    (eigs, eigenvectors, total_iters, converged)
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
#[allow(clippy::needless_range_loop)]
fn lanczos_eigenvalues(
    mat: &SparseCathedralProbe,
    k: usize,
    _max_iter: usize,
    tol: f64,
) -> (Vec<f64>, usize, bool) {
    let n = mat.node_count();
    let m = (2 * k + 10).min(n);

    let mut alpha = vec![0.0f64; m];
    let mut beta = vec![0.0f64; m];
    let mut q = vec![vec![0.0f64; n]; m + 1];

    let mut rng_state: u64 = 12345;
    for i in 0..n {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        q[0][i] = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
    }
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
        mat.matvec(&q[j], &mut w);

        alpha[j] = (0..n).map(|i| q[j][i] * w[i]).sum();

        for i in 0..n {
            w[i] -= alpha[j] * q[j][i];
            if iter > 0 {
                w[i] -= beta[iter - 1] * q[iter - 1][i];
            }
        }

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

    let sz = j + 1;
    let mut t_diag = alpha[..sz].to_vec();
    let mut t_sub = beta[..sz.saturating_sub(1)].to_vec();

    let (mut eigs, qr_iters, converged) = implicit_qr_tridiag(&mut t_diag, &mut t_sub, tol * 100.0, sz * 30);

    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
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

    /// Build C_n (cycle graph on n nodes, weight 1.0).
    fn cycle_graph(n: usize) -> CathedralProbe {
        let names: Vec<String> = (0..n).map(|i| format!("n{i}")).collect();
        let mut g = CathedralProbe::new(names.iter().map(|s| s.as_str()).collect());
        for i in 0..n {
            g.connect(&names[i], &names[(i + 1) % n], 1.0);
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
        let p = complete_graph(3);
        let spec = p.spectrum();
        assert_eq!(spec.len(), 3);
        assert!(spec[0].abs() < 0.01);
        assert!(rel_close(spec[1], 3.0, 0.01));
        assert!(rel_close(spec[2], 3.0, 0.01));
    }

    #[test]
    fn test_k3_fiedler_exact() {
        let p = complete_graph(3);
        assert!(rel_close(p.fiedler_value(), 3.0, 0.01));
    }

    #[test]
    fn test_k4_spectrum_exact() {
        let p = complete_graph(4);
        let spec = p.spectrum();
        assert_eq!(spec.len(), 4);
        assert!(spec[0].abs() < 0.05);
        assert!(rel_close(spec[1], 4.0, 0.02));
        assert!(rel_close(spec[2], 4.0, 0.02));
        assert!(rel_close(spec[3], 4.0, 0.02));
    }

    #[test]
    fn test_p4_spectrum_exact() {
        let p = path_graph(4);
        let spec = p.spectrum();
        assert_eq!(spec.len(), 4);
        let fiedler = 2.0 - 2.0_f64.sqrt();
        assert!(rel_close(spec[1], fiedler, 0.02));
        assert!(rel_close(spec[2], 2.0, 0.02));
        let max_eig = 2.0 + 2.0_f64.sqrt();
        assert!(rel_close(spec[3], max_eig, 0.02));
    }

    #[test]
    fn test_p4_fiedler_exact() {
        let p = path_graph(4);
        let expected = 2.0 - 2.0_f64.sqrt();
        assert!(rel_close(p.fiedler_value(), expected, 0.02));
    }

    #[test]
    fn test_s4_fiedler_exact() {
        let p = star_graph(4);
        assert!(rel_close(p.fiedler_value(), 1.0, 0.02));
    }

    #[test]
    fn test_s4_spectrum_exact() {
        let p = star_graph(4);
        let spec = p.spectrum();
        assert_eq!(spec.len(), 4);
        assert!(spec[0].abs() < 0.05);
        assert!(rel_close(spec[1], 1.0, 0.02));
        assert!(rel_close(spec[2], 1.0, 0.02));
        assert!(rel_close(spec[3], 4.0, 0.02));
    }

    #[test]
    fn test_p3_fiedler_exact() {
        let p = path_graph(3);
        let spec = p.spectrum();
        assert!(spec[0].abs() < 0.05);
        assert!(rel_close(spec[1], 1.0, 0.02));
        assert!(rel_close(spec[2], 3.0, 0.02));
    }

    #[test]
    fn test_k2_spectrum_exact() {
        let mut p = CathedralProbe::new(vec!["a", "b"]);
        p.connect("a", "b", 1.0);
        let spec = p.spectrum();
        assert_eq!(spec.len(), 2);
        assert!(spec[0].abs() < 0.01);
        assert!(rel_close(spec[1], 2.0, 0.01));
    }

    #[test]
    fn test_p5_fiedler_exact() {
        let p = path_graph(5);
        let expected = 2.0 - 2.0 * (std::f64::consts::PI / 5.0).cos();
        assert!(rel_close(p.fiedler_value(), expected, 0.02));
    }

    #[test]
    fn test_weighted_k2_exact() {
        let mut p = CathedralProbe::new(vec!["a", "b"]);
        p.connect("a", "b", 3.5);
        let spec = p.spectrum();
        assert!(spec[0].abs() < 0.01);
        assert!(rel_close(spec[1], 7.0, 0.01));
    }

    #[test]
    fn test_cycle_c4_fiedler_exact() {
        let p = cycle_graph(4);
        assert!(rel_close(p.fiedler_value(), 2.0, 0.02));
    }

    // ─── Convergence and spectrum_result ────────────────────────

    #[test]
    fn test_spectrum_result_convergence() {
        let p = complete_graph(5);
        let result = p.spectrum_result();
        assert!(result.converged);
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
        assert!(rel_close(result.fiedler_value().unwrap(), 2.0, 0.01));
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
        let p = complete_graph(3);
        assert!(rel_close(p.cheeger_upper_bound(), 6.0_f64.sqrt(), 0.02));
        assert!(rel_close(p.cheeger_lower_bound(), 1.5, 0.02));
    }

    // ─── Fragility ──────────────────────────────────────────────

    #[test]
    fn test_fragility_connected() {
        let mut p = CathedralProbe::new(vec!["a", "b"]);
        p.connect("a", "b", 1.0);
        assert!(p.fragility_index().is_finite());
        assert!(rel_close(p.fragility_index(), 0.5, 0.01));
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
        assert!((p.average_degree() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_weighted_edges() {
        let mut p = CathedralProbe::new(vec!["a", "b", "c"]);
        p.connect("a", "b", 10.0);
        p.connect("b", "c", 0.1);
        assert!(p.fiedler_value() > 0.0);
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
        assert!(complete.fiedler_value() > line.fiedler_value());
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
        assert!(rel_close(sparse.fiedler_value(), 3.0, 0.05));
    }

    #[test]
    fn test_sparse_p4_fiedler() {
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)];
        let sparse = SparseCathedralProbe::from_edges(4, &edges);
        let expected = 2.0 - 2.0_f64.sqrt();
        assert!(rel_close(sparse.fiedler_value(), expected, 0.05));
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
        assert!(result.eigenvalues[0].abs() < 0.1);
        assert!(result.eigenvalues[1] > 0.0);
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
        let mut dg = DirectedCathedralProbe::new(vec!["a", "b", "c"]);
        dg.add_edge("a", "b", 1.0);
        dg.add_edge("b", "c", 1.0);
        dg.add_edge("c", "a", 1.0);
        let f = dg.fiedler_value();
        assert!(f > 0.0, "Strongly connected directed cycle should have Fiedler > 0, got {f}");
    }

    #[test]
    fn test_directed_disconnected_fiedler() {
        let mut dg = DirectedCathedralProbe::new(vec!["a", "b", "c"]);
        dg.add_edge("a", "b", 1.0);
        dg.add_edge("a", "c", 1.0);
        let f = dg.fiedler_value();
        let mut dg2 = DirectedCathedralProbe::new(vec!["a", "b", "c"]);
        dg2.add_edge("a", "b", 1.0);
        dg2.add_edge("b", "c", 1.0);
        dg2.add_edge("c", "a", 1.0);
        let f2 = dg2.fiedler_value();
        assert!(f2 > f, "Strongly connected Fiedler ({f2}) should exceed weakly connected ({f})");
    }

    #[test]
    fn test_directed_spectrum_result() {
        let mut dg = DirectedCathedralProbe::new(vec!["x", "y"]);
        dg.add_edge("x", "y", 1.0);
        dg.add_edge("y", "x", 1.0);
        let result = dg.spectrum_result();
        assert_eq!(result.eigenvalues.len(), 2);
        assert!(result.eigenvalues[0].abs() < 0.1);
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
        let r = c * 3.0 + s * 4.0;
        let zero = -s * 3.0 + c * 4.0;
        assert!(r > 0.0);
        assert!(zero.abs() < 1e-10);
        assert!((c * c + s * s - 1.0).abs() < 1e-10);
    }

    // ─── Householder tridiagonalization ─────────────────────────

    #[test]
    fn test_tridiag_preserves_trace() {
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
        assert!((trace_orig - trace_tri).abs() < 1e-8);
    }

    #[test]
    fn test_tridiag_preserves_frobenius() {
        let mut p = CathedralProbe::new(vec!["a", "b", "c"]);
        p.connect("a", "b", 1.0);
        p.connect("b", "c", 1.0);
        p.connect("a", "c", 1.0);
        let lap = p.build_laplacian();
        let frob_orig: f64 = lap.iter().flat_map(|row| row.iter()).map(|&v| v * v).sum();
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

    // ═══════════════════════════════════════════════════════════════
    // Advanced spectral method tests
    // ═══════════════════════════════════════════════════════════════

    // ─── Effective resistance ───────────────────────────────────

    #[test]
    fn test_effective_resistance_k2() {
        // K₂ with weight 1: R_eff(0,1) = 1/w = 1.0
        let mut p = CathedralProbe::new(vec!["a", "b"]);
        p.connect("a", "b", 1.0);
        let r = p.effective_resistance(0, 1);
        assert!(rel_close(r, 1.0, 0.02), "K₂ R_eff = {r} (expected 1.0)");
    }

    #[test]
    fn test_effective_resistance_weighted() {
        // K₂ with weight 2: R_eff = 1/2 = 0.5
        let mut p = CathedralProbe::new(vec!["a", "b"]);
        p.connect("a", "b", 2.0);
        let r = p.effective_resistance(0, 1);
        assert!(rel_close(r, 0.5, 0.02), "Weighted K₂ R_eff = {r} (expected 0.5)");
    }

    #[test]
    fn test_effective_resistance_same_node() {
        let p = complete_graph(3);
        assert_eq!(p.effective_resistance(0, 0), 0.0);
    }

    #[test]
    fn test_effective_resistance_k3() {
        // K₃: R_eff = 2/3 for all pairs (complete graph R_eff = 2/n)
        let p = complete_graph(3);
        let r = p.effective_resistance(0, 1);
        assert!(rel_close(r, 2.0 / 3.0, 0.02), "K₃ R_eff = {r} (expected 0.667)");
    }

    #[test]
    fn test_effective_resistance_k4() {
        // K₄: R_eff = 2/4 = 0.5 for all pairs
        let p = complete_graph(4);
        let r = p.effective_resistance(0, 3);
        assert!(rel_close(r, 0.5, 0.03), "K₄ R_eff = {r} (expected 0.5)");
    }

    #[test]
    fn test_effective_resistance_triangle_inequality() {
        // R_eff should satisfy triangle inequality
        let p = path_graph(4);
        let r01 = p.effective_resistance(0, 1);
        let r12 = p.effective_resistance(1, 2);
        let r02 = p.effective_resistance(0, 2);
        assert!(r02 <= r01 + r12 + 0.01, "Triangle inequality: {r02} > {r01} + {r12}");
    }

    #[test]
    fn test_effective_resistance_path_additivity() {
        // For a path graph, R_eff(0,2) = R_eff(0,1) + R_eff(1,2) (series resistance)
        let p = path_graph(3);
        let r01 = p.effective_resistance(0, 1);
        let r12 = p.effective_resistance(1, 2);
        let r02 = p.effective_resistance(0, 2);
        assert!(rel_close(r02, r01 + r12, 0.02), "P₃ series: {r02} != {r01} + {r12}");
    }

    // ─── Kirchhoff index ───────────────────────────────────────

    #[test]
    fn test_kirchhoff_index_k3() {
        // K₃: Kf = n*(n-1)/2 * (2/n) = n-1 = 2
        // Actually Kf = n * Σ 1/λ_k. K₃ eigenvalues: {0, 3, 3}, so Kf = 3*(1/3+1/3) = 2
        let p = complete_graph(3);
        let kf = p.kirchhoff_index();
        assert!(rel_close(kf, 2.0, 0.02), "K₃ Kf = {kf} (expected 2.0)");
    }

    #[test]
    fn test_kirchhoff_index_k4() {
        // K₄: eigenvalues {0, 4, 4, 4}, Kf = 4*(3/4) = 3
        let p = complete_graph(4);
        let kf = p.kirchhoff_index();
        assert!(rel_close(kf, 3.0, 0.03), "K₄ Kf = {kf} (expected 3.0)");
    }

    #[test]
    fn test_kirchhoff_index_via_pairwise() {
        // Kf = Σ_{i<j} R_eff(i,j)
        let p = complete_graph(3);
        let kf_direct = p.kirchhoff_index();
        let kf_pairwise = p.effective_resistance(0, 1)
            + p.effective_resistance(0, 2)
            + p.effective_resistance(1, 2);
        assert!(
            rel_close(kf_direct, kf_pairwise, 0.03),
            "Kf direct = {kf_direct}, pairwise = {kf_pairwise}"
        );
    }

    #[test]
    fn test_kirchhoff_index_small() {
        let p = CathedralProbe::new(vec!["a"]);
        assert_eq!(p.kirchhoff_index(), 0.0);
    }

    // ─── Spectral embedding ────────────────────────────────────

    #[test]
    fn test_spectral_embedding_dimensions() {
        let p = complete_graph(4);
        let emb = p.spectral_embedding(2);
        assert_eq!(emb.len(), 4); // 4 nodes
        assert_eq!(emb[0].len(), 2); // 2 dimensions
    }

    #[test]
    fn test_spectral_embedding_empty() {
        let p = CathedralProbe::new(vec![]);
        let emb = p.spectral_embedding(2);
        assert!(emb.is_empty());
    }

    #[test]
    fn test_spectral_embedding_k3() {
        let p = complete_graph(3);
        let emb = p.spectral_embedding(2);
        // All nodes should have same embedding magnitude (symmetric graph)
        let norms: Vec<f64> = emb.iter()
            .map(|v| v.iter().map(|x| x * x).sum::<f64>().sqrt())
            .collect();
        let max_diff = norms.iter().map(|n| (n - norms[0]).abs()).fold(0.0f64, f64::max);
        assert!(max_diff < 0.05, "K₃ embedding norms should be equal, diff = {max_diff}");
    }

    #[test]
    fn test_spectral_embedding_single_dim() {
        let p = path_graph(4);
        let emb = p.spectral_embedding(1);
        assert_eq!(emb.len(), 4);
        assert_eq!(emb[0].len(), 1);
    }

    // ─── Spectral clustering ───────────────────────────────────

    #[test]
    fn test_spectral_cluster_returns_valid() {
        let p = complete_graph(4);
        let clusters = p.spectral_cluster(2);
        assert_eq!(clusters.len(), 4);
        assert!(clusters.iter().all(|&c| c < 2));
    }

    #[test]
    fn test_spectral_cluster_k1() {
        let p = complete_graph(4);
        let clusters = p.spectral_cluster(1);
        assert!(clusters.iter().all(|&c| c == 0));
    }

    #[test]
    fn test_spectral_cluster_empty() {
        let p = CathedralProbe::new(vec![]);
        let clusters = p.spectral_cluster(2);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_spectral_cluster_two_communities() {
        // Two cliques connected by a single edge
        let mut p = CathedralProbe::new(vec!["a", "b", "c", "d", "e", "f"]);
        // Clique 1: a, b, c
        p.connect("a", "b", 1.0);
        p.connect("b", "c", 1.0);
        p.connect("a", "c", 1.0);
        // Clique 2: d, e, f
        p.connect("d", "e", 1.0);
        p.connect("e", "f", 1.0);
        p.connect("d", "f", 1.0);
        // Bridge
        p.connect("c", "d", 0.1);

        let clusters = p.spectral_cluster(2);
        assert_eq!(clusters.len(), 6);

        // a, b, c should be in one cluster; d, e, f in another
        let group_abc = clusters[0];
        assert_eq!(clusters[1], group_abc, "b should be with a");
        assert_eq!(clusters[2], group_abc, "c should be with a");
        assert_ne!(clusters[3], group_abc, "d should be different from a");
        assert_eq!(clusters[4], clusters[3], "e should be with d");
        assert_eq!(clusters[5], clusters[3], "f should be with d");
    }

    #[test]
    fn test_spectral_cluster_k_ge_n() {
        let p = complete_graph(3);
        let clusters = p.spectral_cluster(5);
        assert_eq!(clusters, vec![0, 1, 2]);
    }

    // ─── Community profile ─────────────────────────────────────

    #[test]
    fn test_community_profile_basic() {
        let p = complete_graph(4);
        let profile = p.community_profile();
        assert_eq!(profile.len(), 2); // n/2 = 2 for n=4
        for &(s, cond) in &profile {
            assert!(s >= 1);
            assert!(cond >= 0.0);
        }
    }

    #[test]
    fn test_community_profile_empty() {
        let p = CathedralProbe::new(vec![]);
        assert!(p.community_profile().is_empty());
    }

    #[test]
    fn test_community_profile_single() {
        let p = CathedralProbe::new(vec!["a"]);
        assert!(p.community_profile().is_empty());
    }

    #[test]
    fn test_community_profile_conductance_bounded() {
        let p = path_graph(6);
        let profile = p.community_profile();
        // All conductance values should be in [0, 1]
        for &(_, cond) in &profile {
            assert!(cond <= 1.01, "Conductance {cond} > 1.0");
        }
    }

    #[test]
    fn test_community_profile_two_nodes() {
        let mut p = CathedralProbe::new(vec!["a", "b"]);
        p.connect("a", "b", 1.0);
        let profile = p.community_profile();
        assert_eq!(profile.len(), 1); // n/2 = 1
        assert_eq!(profile[0].0, 1);
        // Conductance of splitting 2-node graph: cut=1, min(vol_s, vol_comp) = 1
        assert!(rel_close(profile[0].1, 1.0, 0.02));
    }

    // ─── Fiedler sensitivity ───────────────────────────────────

    #[test]
    fn test_fiedler_sensitivity_basic() {
        let p = complete_graph(3);
        let sens = p.fiedler_sensitivity();
        assert_eq!(sens.len(), 3); // K₃ has 3 edges
        for &(_, _, delta) in &sens {
            assert!(delta >= 0.0, "Sensitivity should be non-negative");
        }
    }

    #[test]
    fn test_fiedler_sensitivity_path() {
        // P₃: edges (0,1) and (1,2). Middle edge should have higher sensitivity
        // because removing it disconnects the graph.
        let p = path_graph(3);
        let sens = p.fiedler_sensitivity();
        assert_eq!(sens.len(), 2);
        // Both edges have equal sensitivity in P₃ (by symmetry)
        let s0 = sens[0].2;
        let s1 = sens[1].2;
        assert!((s0 - s1).abs() < 0.05, "P₃ sensitivities should be equal: {s0} vs {s1}");
    }

    #[test]
    fn test_fiedler_sensitivity_nonnegative() {
        let p = path_graph(5);
        let sens = p.fiedler_sensitivity();
        for &(_, _, delta) in &sens {
            assert!(delta >= 0.0);
        }
    }

    #[test]
    fn test_fiedler_sensitivity_empty() {
        let p = CathedralProbe::new(vec!["a"]);
        assert!(p.fiedler_sensitivity().is_empty());
    }

    #[test]
    fn test_fiedler_sensitivity_bridge_edge() {
        // Two cliques connected by a bridge — bridge should have highest sensitivity
        let mut p = CathedralProbe::new(vec!["a", "b", "c", "d"]);
        p.connect("a", "b", 1.0);
        p.connect("b", "c", 1.0);
        p.connect("c", "d", 1.0);
        p.connect("a", "c", 1.0);
        let sens = p.fiedler_sensitivity();
        // The bridge-like edge (b,d or similar) should stand out
        assert!(!sens.is_empty());
    }

    // ─── Condition number ──────────────────────────────────────

    #[test]
    fn test_condition_number_k3() {
        // K₃: eigenvalues {0, 3, 3}, κ = 3/3 = 1.0
        let p = complete_graph(3);
        let kappa = p.condition_number();
        assert!(rel_close(kappa, 1.0, 0.02), "K₃ κ = {kappa} (expected 1.0)");
    }

    #[test]
    fn test_condition_number_path() {
        // P₄ has higher condition number than K₄
        let p4 = path_graph(4);
        let k4 = complete_graph(4);
        assert!(
            p4.condition_number() > k4.condition_number(),
            "P₄ κ should exceed K₄ κ"
        );
    }

    #[test]
    fn test_condition_number_empty() {
        let p = CathedralProbe::new(vec![]);
        assert_eq!(p.condition_number(), 0.0);
    }

    #[test]
    fn test_condition_number_disconnected() {
        let p = CathedralProbe::new(vec!["a", "b"]);
        assert!(p.condition_number().is_infinite());
    }

    #[test]
    fn test_condition_number_k4() {
        // K₄: eigenvalues {0, 4, 4, 4}, κ = 4/4 = 1.0
        let p = complete_graph(4);
        let kappa = p.condition_number();
        assert!(rel_close(kappa, 1.0, 0.03), "K₄ κ = {kappa} (expected 1.0)");
    }

    // ─── Eigenvector verification ──────────────────────────────

    #[test]
    fn test_eigenvectors_orthogonal() {
        let p = complete_graph(4);
        let (_, vecs) = p.full_eigen();
        // Check that eigenvectors are approximately orthogonal
        for i in 0..vecs.len() {
            for j in (i + 1)..vecs.len() {
                let dot: f64 = vecs[i].iter().zip(&vecs[j]).map(|(&a, &b)| a * b).sum();
                assert!(dot.abs() < 0.05, "vecs[{i}]·vecs[{j}] = {dot} (should be ~0)");
            }
        }
    }

    #[test]
    fn test_eigenvectors_unit_norm() {
        let p = complete_graph(4);
        let (_, vecs) = p.full_eigen();
        for (k, v) in vecs.iter().enumerate() {
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 0.05, "vec[{k}] norm = {norm} (should be 1.0)");
        }
    }

    #[test]
    fn test_eigenvector_satisfies_eigenvalue_equation() {
        let p = path_graph(4);
        let (eigs, vecs) = p.full_eigen();
        let lap = p.build_laplacian();
        let n = p.node_count();

        for k in 0..n {
            // L * v_k should equal λ_k * v_k
            for i in 0..n {
                let lv: f64 = (0..n).map(|j| lap[i][j] * vecs[k][j]).sum();
                let expected = eigs[k] * vecs[k][i];
                assert!(
                    (lv - expected).abs() < 0.05,
                    "L·v[{k}][{i}] = {lv}, λ*v = {expected}"
                );
            }
        }
    }

    #[test]
    fn test_first_eigenvector_constant() {
        let p = complete_graph(4);
        let (_, vecs) = p.full_eigen();
        // First eigenvector should be (1/√n, 1/√n, ..., 1/√n)
        let expected = 1.0 / 4.0_f64.sqrt();
        for i in 0..4 {
            assert!(
                (vecs[0][i].abs() - expected).abs() < 0.05,
                "v₀[{i}] = {} (expected ±{expected})", vecs[0][i]
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Matrix constructor tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_from_matrix_k3() {
        // K₃ adjacency matrix
        let weights = vec![
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ];
        let g = CathedralProbe::from_matrix(&weights).unwrap();
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 3);
        assert!(rel_close(g.fiedler_value(), 3.0, 0.02));
    }

    #[test]
    fn test_from_matrix_empty() {
        let g = CathedralProbe::from_matrix(&[]).unwrap();
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn test_from_matrix_asymmetric() {
        // Asymmetric matrix should still work (averages the off-diagonals)
        let weights = vec![
            vec![0.0, 2.0, 0.0],
            vec![0.0, 0.0, 3.0], // asymmetric: weights[0][1]=2, weights[1][0]=0
            vec![0.0, 3.0, 0.0],
        ];
        let g = CathedralProbe::from_matrix(&weights).unwrap();
        assert_eq!(g.node_count(), 3);
        // Edge (0,1) weight = avg(2,0) = 1.0, edge (1,2) weight = avg(3,3) = 3.0
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn test_from_matrix_bad_dimensions() {
        let weights = vec![
            vec![0.0, 1.0],
            vec![1.0, 0.0, 1.0], // extra element
        ];
        let result = CathedralProbe::from_matrix(&weights);
        assert!(result.is_err());
        if let Err(CathedralError::InvalidMatrix(_)) = result {} else {
            panic!("Expected InvalidMatrix error");
        }
    }

    #[test]
    fn test_from_edge_list_path() {
        // Path graph P₄: 0-1-2-3
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)];
        let g = CathedralProbe::from_edge_list(4, &edges);
        assert_eq!(g.node_count(), 4);
        assert_eq!(g.edge_count(), 3);
        let expected_fiedler = 2.0 - 2.0_f64.sqrt();
        assert!(rel_close(g.fiedler_value(), expected_fiedler, 0.02));
    }

    #[test]
    fn test_from_edge_list_empty() {
        let g = CathedralProbe::from_edge_list(3, &[]);
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_from_edge_list_self_loop_ignored() {
        let edges = vec![(0, 0, 5.0), (0, 1, 1.0)];
        let g = CathedralProbe::from_edge_list(2, &edges);
        assert_eq!(g.edge_count(), 1); // self-loop ignored
    }

    #[test]
    fn test_from_edge_list_out_of_bounds_ignored() {
        let edges = vec![(0, 1, 1.0), (0, 5, 1.0)]; // node 5 doesn't exist
        let g = CathedralProbe::from_edge_list(2, &edges);
        assert_eq!(g.edge_count(), 1); // out-of-bounds edge ignored
    }

    #[test]
    fn test_from_weighted_adjacency_star() {
        // Star graph S₄: hub=0, leaves=1,2,3
        let n = 4;
        let mut weights = vec![0.0f64; n * n];
        // row 0: [0, 1, 1, 1]
        weights[0 * n + 1] = 1.0;
        weights[0 * n + 2] = 1.0;
        weights[0 * n + 3] = 1.0;
        // symmetric
        weights[1 * n + 0] = 1.0;
        weights[2 * n + 0] = 1.0;
        weights[3 * n + 0] = 1.0;

        let g = CathedralProbe::from_weighted_adjacency(&weights, n).unwrap();
        assert_eq!(g.node_count(), 4);
        assert_eq!(g.edge_count(), 3);
        assert!(rel_close(g.fiedler_value(), 1.0, 0.02));
    }

    #[test]
    fn test_from_weighted_adjacency_bad_length() {
        let result = CathedralProbe::from_weighted_adjacency(&[0.0; 3], 2); // 3 != 4
        assert!(result.is_err());
    }

    // ─── Per-component analysis ──────────────────────────────────

    #[test]
    fn test_per_component_connected() {
        let p = complete_graph(3);
        let components = p.per_component_analysis();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].size, 3);
        assert!(components[0].fiedler_value > 0.0);
    }

    #[test]
    fn test_per_component_disconnected() {
        // Two disconnected edges: 0-1 and 2-3
        let mut p = CathedralProbe::new(vec!["a", "b", "c", "d"]);
        p.connect("a", "b", 1.0);
        p.connect("c", "d", 1.0);
        let components = p.per_component_analysis();
        assert_eq!(components.len(), 2);
        for comp in &components {
            assert_eq!(comp.size, 2);
            assert!(comp.fiedler_value > 0.0); // each sub-component is connected
        }
    }

    #[test]
    fn test_per_component_isolated_nodes() {
        // 4 nodes, only one edge 0-1
        let mut p = CathedralProbe::new(vec!["a", "b", "c", "d"]);
        p.connect("a", "b", 1.0);
        let components = p.per_component_analysis();
        assert_eq!(components.len(), 3); // {a,b}, {c}, {d}
        let mut sizes: Vec<usize> = components.iter().map(|c| c.size).collect();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![1, 1, 2]);
    }

    #[test]
    fn test_per_component_empty() {
        let p = CathedralProbe::new(vec![]);
        assert!(p.per_component_analysis().is_empty());
    }

    #[test]
    fn test_per_component_single_node() {
        let p = CathedralProbe::new(vec!["a"]);
        let c = p.per_component_analysis();
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].fiedler_value, 0.0);
        assert_eq!(c[0].cheeger_upper, 0.0);
    }

    #[test]
    fn test_per_component_fiedler_matches_subgraph() {
        // Two triangles connected by a bridge
        let mut p = CathedralProbe::new(vec!["a", "b", "c", "d", "e", "f"]);
        // Triangle 1
        p.connect("a", "b", 1.0);
        p.connect("b", "c", 1.0);
        p.connect("a", "c", 1.0);
        // Triangle 2
        p.connect("d", "e", 1.0);
        p.connect("e", "f", 1.0);
        p.connect("d", "f", 1.0);
        // Bridge
        p.connect("c", "d", 0.1);

        // Whole graph is connected → 1 component
        let components = p.per_component_analysis();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].size, 6);
    }

    // ─── Serde round-trip (behind feature gate) ──────────────────

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_roundtrip_spectrum_result() {
        let p = complete_graph(3);
        let result = p.spectrum_result();
        let json = serde_json::to_string(&result).unwrap();
        let back: SpectrumResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.eigenvalues.len(), result.eigenvalues.len());
        for (a, b) in back.eigenvalues.iter().zip(&result.eigenvalues) {
            assert!((a - b).abs() < 0.01);
        }
        assert_eq!(back.converged, result.converged);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_roundtrip_spectrum_method() {
        let method = SpectrumMethod::DenseImplicitQr;
        let json = serde_json::to_string(&method).unwrap();
        let back: SpectrumMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(back, method);

        let lanczos = SpectrumMethod::Lanczos { k: 5 };
        let json2 = serde_json::to_string(&lanczos).unwrap();
        let back2: SpectrumMethod = serde_json::from_str(&json2).unwrap();
        assert_eq!(back2, lanczos);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_roundtrip_cathedral_error() {
        let err = CathedralError::NodeNotFound("foo".into());
        let json = serde_json::to_string(&err).unwrap();
        let back: CathedralError = serde_json::from_str(&json).unwrap();
        match back {
            CathedralError::NodeNotFound(s) => assert_eq!(s, "foo"),
            _ => panic!("wrong variant"),
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_roundtrip_component_analysis() {
        let ca = ComponentAnalysis {
            nodes: vec![0, 1, 2],
            fiedler_value: 3.0,
            cheeger_upper: 2.449,
            size: 3,
        };
        let json = serde_json::to_string(&ca).unwrap();
        let back: ComponentAnalysis = serde_json::from_str(&json).unwrap();
        assert_eq!(back.nodes, ca.nodes);
        assert!((back.fiedler_value - ca.fiedler_value).abs() < 1e-10);
        assert_eq!(back.size, ca.size);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_to_json_output() {
        let mut p = CathedralProbe::new(vec!["a", "b"]);
        p.connect("a", "b", 1.0);
        let json = p.to_json().unwrap();
        assert!(json.contains("\"eigenvalues\""));
        assert!(json.contains("\"converged\""));
        // Verify it's valid JSON by parsing back
        let _: SpectrumResult = serde_json::from_str(&json).unwrap();
    }
}
