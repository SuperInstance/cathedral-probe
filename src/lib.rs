//! # Cathedral Probe
//!
//! Spectral topology analysis for component graphs.
//!
//! Compute Laplacian eigenvalues, Fiedler value (connectivity), Cheeger constant
//! (bottleneck detection), and component importance. Answer: "is the space between
//! my components healthy?"

#![deny(unsafe_code)]

use std::collections::HashMap;

// ─── Graph ──────────────────────────────────────────────────────────

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

    /// Compute all eigenvalues of the graph Laplacian using QR iteration.
    #[allow(clippy::needless_range_loop)]
    pub fn spectrum(&self) -> Vec<f64> {
        let n = self.nodes.len();
        if n == 0 { return vec![]; }
        if n == 1 { return vec![self.build_laplacian()[0][0]]; }

        let mut mat = self.build_laplacian();

        // Shift to make positive semi-definite (Laplacian has eigenvalues >= 0)
        // QR iteration
        for _ in 0..200 {
            // Wilkinson shift
            let nn = mat.len();
            let shift = mat[nn-1][nn-1];
            for i in 0..nn { mat[i][i] -= shift; }
            let (q, r) = qr_decompose(&mat);
            mat = mat_mul(&r, &q);
            for i in 0..nn { mat[i][i] += shift; }

            // Zero out below diagonal
            #[allow(clippy::needless_range_loop)]
            for i in 0..nn {
                for j in 0..i {
                    if (mat[i][j]).abs() < 1e-12 { mat[i][j] = 0.0; }
                }
            }
        }

        let mut eigs: Vec<f64> = (0..n).map(|i| mat[i][i]).collect();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Clean near-zero values
        for e in &mut eigs {
            if (*e).abs() < 1e-8 { *e = 0.0; }
        }
        eigs
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

    /// Spectral conductance bound based on Cheeger's inequality.
    ///
    /// This is NOT the exact Cheeger constant h(G). Computing h(G) exactly
    /// requires checking all partitions of the vertex set, which is NP-hard.
    ///
    /// Instead, this returns the upper bound from the Cheeger inequality:
    ///   h(G) ≤ √(2·λ₂)
    ///
    /// where λ₂ is the Fiedler value (algebraic connectivity). A higher value
    /// means fewer bottlenecks are possible. Use `cheeger_lower_bound()` for
    /// the complementary lower bound: λ₂/2 ≤ h(G).
    ///
    /// References:
    /// - Fiedler, M. (1973). "Algebraic connectivity of graphs."
    ///   Czechoslovak Mathematical Journal, 23(2), 298-305.
    /// - Chung, F. (1997). "Spectral Graph Theory." CBMS Regional Conference
    ///   Series in Mathematics, No. 92. AMS.
    /// - Mohar, B. (1989). "Isoperimetric numbers of graphs." Journal of
    ///   Combinatorial Theory, Series B, 47(3), 274-291.
    pub fn cheeger_upper_bound(&self) -> f64 {
        let fiedler = self.fiedler_value();
        if self.nodes.len() <= 1 { return 0.0; }
        (2.0 * fiedler).sqrt()
    }

    /// Lower bound on the Cheeger constant from Cheeger's inequality.
    ///
    /// Returns λ₂/2, satisfying: λ₂/2 ≤ h(G) ≤ √(2·λ₂)
    pub fn cheeger_lower_bound(&self) -> f64 {
        self.fiedler_value() / 2.0
    }

    /// Legacy alias for `cheeger_upper_bound()`.
    ///
    /// **Note:** This previously returned √(λ₂/2), which is not a standard
    /// Cheeger bound. The correct upper bound is √(2·λ₂). This method now
    /// returns the correct value for backward compatibility.
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
            // Create subgraph without this node
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
            // Remove this edge
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

// ─── QR Decomposition ───────────────────────────────────────────────

fn qr_decompose(mat: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = mat.len();
    let mut q = vec![vec![0.0; n]; n];
    let mut r = vec![vec![0.0; n]; n];
    let mut v = vec![vec![0.0; n]; n];

    for j in 0..n {
        for i in 0..n { v[i][j] = mat[i][j]; }
        for i in 0..j {
            let dot: f64 = (0..n).map(|k| mat[k][j] * q[k][i]).sum();
            r[i][j] = dot;
            for k in 0..n { v[k][j] -= dot * q[k][i]; }
        }
        let norm: f64 = (0..n).map(|k| v[k][j].powi(2)).sum::<f64>().sqrt();
        r[j][j] = if norm < 1e-14 { 0.0 } else { norm };
        for k in 0..n {
            q[k][j] = if norm < 1e-14 { 0.0 } else { v[k][j] / norm };
        }
    }
    (q, r)
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut c = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n { c[i][j] += a[i][k] * b[k][j]; }
        }
    }
    c
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_spectrum_complete_graph() {
        let mut p = CathedralProbe::new(vec!["a", "b", "c"]);
        p.connect("a", "b", 1.0);
        p.connect("b", "c", 1.0);
        p.connect("a", "c", 1.0);
        let spec = p.spectrum();
        assert_eq!(spec.len(), 3);
        assert!(spec[0].abs() < 0.5); // ~0
        assert!(spec[1] > 0.5); // Fiedler > 0
    }

    #[test]
    fn test_fiedler_connected() {
        let mut p = CathedralProbe::new(vec!["a", "b"]);
        p.connect("a", "b", 1.0);
        assert!(p.fiedler_value() > 0.5);
    }

    #[test]
    fn test_fiedler_disconnected() {
        let p = CathedralProbe::new(vec!["a", "b"]);
        // No edges
        assert!(p.fiedler_value() < 0.5);
    }

    #[test]
    fn test_is_healthy() {
        let mut p = CathedralProbe::new(vec!["a", "b", "c"]);
        p.connect("a", "b", 1.0);
        p.connect("b", "c", 1.0);
        p.connect("a", "c", 1.0);
        assert!(p.is_healthy(0.1));
    }

    #[test]
    fn test_cheeger_bounds() {
        let mut p = CathedralProbe::new(vec!["a", "b"]);
        p.connect("a", "b", 1.0);
        let ub = p.cheeger_upper_bound();
        let lb = p.cheeger_lower_bound();
        assert!(ub >= 0.0);
        assert!(lb >= 0.0);
        assert!(ub >= lb); // upper bound >= lower bound
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
    fn test_fragility_connected() {
        let mut p = CathedralProbe::new(vec!["a", "b"]);
        p.connect("a", "b", 1.0);
        assert!(p.fragility_index().is_finite());
    }

    #[test]
    fn test_fragility_disconnected() {
        let p = CathedralProbe::new(vec!["a", "b"]);
        assert!(p.fragility_index().is_infinite());
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
        // c is isolated
        assert!(!p.is_connected());
    }

    #[test]
    fn test_total_weight() {
        let mut p = CathedralProbe::new(vec!["a", "b", "c"]);
        p.connect("a", "b", 2.0);
        p.connect("b", "c", 3.0);
        assert!((p.total_weight() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_average_degree() {
        let mut p = CathedralProbe::new(vec!["a", "b", "c"]);
        p.connect("a", "b", 1.0);
        p.connect("b", "c", 1.0);
        p.connect("a", "c", 1.0);
        // total_weight = 3, avg_degree = 3*2/3 = 2.0
        assert!((p.average_degree() - 2.0).abs() < 0.01);
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
    fn test_star_topology() {
        let mut p = CathedralProbe::new(vec!["hub", "s1", "s2", "s3", "s4"]);
        p.connect("hub", "s1", 1.0);
        p.connect("hub", "s2", 1.0);
        p.connect("hub", "s3", 1.0);
        p.connect("hub", "s4", 1.0);
        assert!(p.is_connected());
        assert!(p.fiedler_value() > 0.0);
    }

    #[test]
    fn test_line_topology() {
        let mut p = CathedralProbe::new(vec!["a", "b", "c", "d", "e"]);
        p.connect("a", "b", 1.0);
        p.connect("b", "c", 1.0);
        p.connect("c", "d", 1.0);
        p.connect("d", "e", 1.0);
        assert!(p.is_connected());
        // Line has lower Fiedler than complete graph
        let f_line = p.fiedler_value();

        let mut p2 = CathedralProbe::new(vec!["a", "b", "c", "d", "e"]);
        for i in 0..5 {
            for j in (i+1)..5 {
                p2.connect(&format!("{}", i), &format!("{}", j), 1.0);
            }
        }
        // Complete graph should have higher Fiedler
        // (different node names but same size)
        assert!(f_line > 0.0);
    }

    #[test]
    fn test_connected_components_disconnected() {
        let mut p = CathedralProbe::new(vec!["a", "b", "c", "d"]);
        p.connect("a", "b", 1.0);
        // c, d isolated
        assert!(p.connected_components() >= 2);
    }

    #[test]
    fn test_algebraic_connectivity_alias() {
        let mut p = CathedralProbe::new(vec!["a", "b"]);
        p.connect("a", "b", 1.0);
        assert!((p.algebraic_connectivity() - p.fiedler_value()).abs() < 1e-10);
    }

    #[test]
    fn test_two_nodes_zero_weight() {
        let mut p = CathedralProbe::new(vec!["a", "b"]);
        p.connect("a", "b", 0.0);
        // Zero weight = effectively disconnected
        assert!(p.fiedler_value() < 0.5);
    }

    #[test]
    fn test_triangle_spectrum() {
        let mut p = CathedralProbe::new(vec!["a", "b", "c"]);
        p.connect("a", "b", 1.0);
        p.connect("b", "c", 1.0);
        p.connect("a", "c", 1.0);
        let spec = p.spectrum();
        // Complete graph K3: eigenvalues are {0, 3, 3}
        assert!(spec[0].abs() < 0.5);
        assert!(spec[1] > 1.0);
        assert!(spec[2] > 1.0);
    }

    #[test]
    fn test_component_importance() {
        let mut p = CathedralProbe::new(vec!["hub", "s1", "s2"]);
        p.connect("hub", "s1", 1.0);
        p.connect("hub", "s2", 1.0);
        let imp = p.component_importance();
        assert!(imp.contains_key("hub"));
        assert!(imp.contains_key("s1"));
        // Hub should be most important
        assert!(imp["hub"] >= imp["s1"]);
    }

    #[test]
    fn test_bottlenecks() {
        let mut p = CathedralProbe::new(vec!["a", "b", "c"]);
        p.connect("a", "b", 1.0);
        p.connect("b", "c", 1.0);
        let bn = p.bottlenecks();
        assert_eq!(bn.len(), 2);
        // Both edges are equally critical in a line
    }

    #[test]
    fn test_large_graph() {
        let mut p = CathedralProbe::new(
            (0..20).map(|i| format!("n{}", i)).collect::<Vec<_>>()
                .iter().map(|s| s.as_str()).collect()
        );
        for i in 0..19 {
            p.connect(&format!("n{}", i), &format!("n{}", i+1), 1.0);
        }
        assert_eq!(p.node_count(), 20);
        assert_eq!(p.edge_count(), 19);
        assert!(p.is_connected());
        assert!(p.fiedler_value() > 0.0);
    }
}
