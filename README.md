# cathedral-probe

**Spectral topology analysis for component graphs.**

Measure the health of the space between your microservices, modules, or any connected system. Compute Laplacian eigenvalues, Fiedler value (connectivity), Cheeger constant (bottleneck detection), and component importance.

```toml
[dependencies]
cathedral-probe = "0.1"
```

## 30-Second Example

```rust
use cathedral_probe::CathedralProbe;

let mut probe = CathedralProbe::new(vec!["web", "api", "db", "cache"]);
probe.connect("web", "api", 1.0);
probe.connect("api", "db", 1.0);
probe.connect("api", "cache", 0.5);

println!("Fiedler value: {:.3}", probe.fiedler_value());
println!("Is healthy: {}", probe.is_healthy(0.1));
println!("Cheeger constant: {:.3}", probe.cheeger_constant());
```

## What It Does

This crate treats your component graph as a mathematical object (a weighted undirected graph) and computes spectral properties of its Laplacian matrix. The "Laplacian" is a matrix that encodes how strongly each component is connected to its neighbors. Its eigenvalues reveal the structure of the space between components.

| Metric | What It Tells You |
|--------|------------------|
| **Fiedler value** | Second-smallest eigenvalue. Higher = better connected. Zero = disconnected. |
| **Cheeger constant** | Bottleneck measure, 0 to 1. Higher = fewer bottlenecks. |
| **Fragility index** | 1 / Fiedler value. Higher = more fragile. Infinity = disconnected. |
| **Component importance** | How much removing each component hurts connectivity. |
| **Bottleneck edges** | Edges whose removal most reduces connectivity. |
| **Spectrum** | All eigenvalues — the full "fingerprint" of your topology. |

## Real-World Use Cases

- **Microservice monitoring** — Is the space between your services healthy or fragmenting?
- **Dependency analysis** — Which packages in your dependency graph are critical?
- **Team communication** — Map who talks to whom, find silos and bottlenecks.
- **Network topology** — Detect when a network is approaching disconnection.
- **Data pipeline health** — Are your ETL stages well-connected or fragile?

## API Reference

```rust
// Create a graph with named components
let mut probe = CathedralProbe::new(vec!["auth", "api", "db", "queue"]);

// Add weighted edges (weight = connection strength)
probe.connect("auth", "api", 1.0);
probe.connect("api", "db", 1.0);

// Spectral analysis
let eigenvalues = probe.spectrum();           // All Laplacian eigenvalues
let fiedler = probe.fiedler_value();           // Connectivity measure
let cheeger = probe.cheeger_constant();        // Bottleneck measure
let fragile = probe.fragility_index();         // 1/fiedler (infinity if disconnected)

// Health check
let healthy = probe.is_healthy(0.1);           // Quick boolean: Fiedler >= threshold?

// Component analysis
let importance = probe.component_importance(); // HashMap<String, f64>
let bottlenecks = probe.bottlenecks();         // Vec<(String, String, f64)>

// Graph properties
let connected = probe.is_connected();          // Fully connected?
let components = probe.connected_components(); // Number of isolated groups
let total = probe.total_weight();              // Sum of edge weights
let avg_deg = probe.average_degree();          // Average weighted degree
```

## How It Works

1. Build the graph Laplacian (degree matrix minus adjacency matrix)
2. Compute eigenvalues via QR iteration with Wilkinson shifts
3. The sorted eigenvalues form the "spectrum" — a fingerprint of the topology
4. The second-smallest eigenvalue (Fiedler value) measures algebraic connectivity
5. Component importance is computed by removing each node and measuring Fiedler drop

Zero dependencies. Works on `no_std` targets with `alloc`.

## License

MIT
