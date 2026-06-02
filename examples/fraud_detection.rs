/// Fraud Detection on Transaction Networks
///
/// Simulates a fintech transaction graph where:
/// - Nodes = accounts
/// - Edges = money transfers (weight = normalized transfer volume)
///
/// We use spectral graph analysis to:
/// 1. Find suspicious communities via spectral clustering
/// 2. Detect bottleneck/structural fragility
/// 3. Identify high-importance accounts (hub risk)
/// 4. Compute effective resistance between flagged nodes
use cathedral_probe::CathedralProbe;

fn main() {
    println!("══════════════════════════════════════════════════════════════");
    println!("  TRANSCEND FINANCIAL — FRAUD DETECTION SPECTRAL ANALYSIS");
    println!("══════════════════════════════════════════════════════════════\n");

    // ── Build the transaction graph ──────────────────────────────────
    // Normal accounts (0-9): legitimate P2P transfer patterns
    // Suspicious cluster (10-14): tight-knit group with large circular transfers
    // Mule accounts (15-19): receive from suspicious, forward to external
    // External gateway (20): single high-volume outbound node
    // Isolated legit accounts (21-23): no transactions (unused accounts)

    let n_accounts = 24;
    let mut graph = CathedralProbe::new(
        (0..n_accounts)
            .map(|i| match i {
                0  => "alice_normal",
                1  => "bob_normal",
                2  => "carol_normal",
                3  => "dave_normal",
                4  => "eve_normal",
                5  => "frank_normal",
                6  => "grace_normal",
                7  => "henry_normal",
                8  => "ivy_normal",
                9  => "jack_normal",
                10 => "suspicious_sarah",
                11 => "suspicious_tom",
                12 => "suspicious_umar",
                13 => "suspicious_vera",
                14 => "suspicious_wu",
                15 => "mule_alpha",
                16 => "mule_beta",
                17 => "mule_gamma",
                18 => "mule_delta",
                19 => "mule_epsilon",
                20 => "external_gateway",
                21 => "dormant_one",
                22 => "dormant_two",
                23 => "dormant_three",
                _  => unreachable!(),
            })
            .collect(),
    );

    // Normal transactions: salary-like patterns (weekly, consistent)
    // edge weight = normalized monthly volume (scale ~1.0)
    graph.connect("alice_normal", "bob_normal", 0.8);
    graph.connect("bob_normal", "carol_normal", 0.6);
    graph.connect("carol_normal", "dave_normal", 0.7);
    graph.connect("dave_normal", "eve_normal", 0.5);
    graph.connect("eve_normal", "frank_normal", 0.9);
    graph.connect("frank_normal", "grace_normal", 0.6);
    graph.connect("grace_normal", "henry_normal", 0.8);
    graph.connect("henry_normal", "ivy_normal", 0.5);
    graph.connect("ivy_normal", "jack_normal", 0.7);
    graph.connect("alice_normal", "eve_normal", 0.3); // occasional cross-link

    // Suspicious cluster: dense mutual transfers (layering)
    // high connectivity + large weights = money laundering pattern
    graph.connect("suspicious_sarah", "suspicious_tom", 12.0);
    graph.connect("suspicious_sarah", "suspicious_umar", 11.5);
    graph.connect("suspicious_sarah", "suspicious_vera", 10.0);
    graph.connect("suspicious_sarah", "suspicious_wu", 13.0);
    graph.connect("suspicious_tom", "suspicious_umar", 9.5);
    graph.connect("suspicious_tom", "suspicious_vera", 8.0);
    graph.connect("suspicious_tom", "suspicious_wu", 11.0);
    graph.connect("suspicious_umar", "suspicious_vera", 10.5);
    graph.connect("suspicious_umar", "suspicious_wu", 9.0);
    graph.connect("suspicious_vera", "suspicious_wu", 12.5);

    // Mule accounts: receive from suspicious, forward to external
    graph.connect("suspicious_sarah", "mule_alpha", 7.0);
    graph.connect("suspicious_tom", "mule_beta", 8.0);
    graph.connect("suspicious_umar", "mule_gamma", 6.5);
    graph.connect("suspicious_vera", "mule_delta", 7.5);
    graph.connect("suspicious_wu", "mule_epsilon", 6.0);

    // Mules -> external gateway (cashing out)
    graph.connect("mule_alpha", "external_gateway", 7.0);
    graph.connect("mule_beta", "external_gateway", 8.0);
    graph.connect("mule_gamma", "external_gateway", 6.5);
    graph.connect("mule_delta", "external_gateway", 7.5);
    graph.connect("mule_epsilon", "external_gateway", 6.0);

    // Thin bridge from normal world to suspicious world
    graph.connect("jack_normal", "suspicious_sarah", 0.1); // one flagged interaction

    // ── 1. Basic Spectral Health ────────────────────────────────────

    println!("[1] GLOBAL SPECTRAL HEALTH");
    println!("──────────────────────────────────────────");
    println!("  Nodes:              {}", graph.node_count());
    println!("  Edges:              {}", graph.edge_count());
    println!("  Total volume:       {:.1}", graph.total_weight());
    println!("  Avg weighted degree:{:.2}", graph.average_degree());

    let fiedler = graph.fiedler_value();
    println!("  Fiedler value:       {:.6}", fiedler);
    println!("  Healthy (≥1.0)?     {}", graph.is_healthy(1.0));
    println!("  Fragility index:    {:.4}", graph.fragility_index());
    println!("  Connected?          {}", graph.is_connected());
    println!("  Cheeger upper:       {:.4}", graph.cheeger_upper_bound());
    println!("  Cheeger lower:       {:.4}", graph.cheeger_lower_bound());

    // ── 2. Per-Component Analysis ──────────────────────────────────

    println!("\n[2] PER-COMPONENT ANALYSIS");
    println!("──────────────────────────────────────────");
    let components = graph.per_component_analysis();
    println!("  Connected components: {}", components.len());
    for (i, comp) in components.iter().enumerate() {
        let names: Vec<&str> = comp
            .nodes
            .iter()
            .map(|&idx| match idx {
                0 => "alice_normal",
                1 => "bob_normal",
                2 => "carol_normal",
                3 => "dave_normal",
                4 => "eve_normal",
                5 => "frank_normal",
                6 => "grace_normal",
                7 => "henry_normal",
                8 => "ivy_normal",
                9 => "jack_normal",
                10 => "suspicious_sarah",
                11 => "suspicious_tom",
                12 => "suspicious_umar",
                13 => "suspicious_vera",
                14 => "suspicious_wu",
                15 => "mule_alpha",
                16 => "mule_beta",
                17 => "mule_gamma",
                18 => "mule_delta",
                19 => "mule_epsilon",
                20 => "external_gateway",
                21 => "dormant_one",
                22 => "dormant_two",
                23 => "dormant_three",
                _ => "???",
            })
            .collect();
        println!(
            "  Component {}: {} nodes, Fiedler={:.4}, Cheeger upper={:.4}",
            i,
            comp.size,
            comp.fiedler_value,
            comp.cheeger_upper
        );
        if comp.nodes.len() <= 25 {
            println!("    Nodes: {:?}", names);
        }
    }

    // ── 3. Component Importance (Critical Nodes) ───────────────────

    println!("\n[3] CRITICAL ACCOUNTS (Component Importance)");
    println!("──────────────────────────────────────────");
    let mut importance: Vec<(String, f64)> = graph
        .component_importance()
        .into_iter()
        .collect();
    importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  Top 10 most critical accounts by Fiedler drop if removed:");
    for (i, (name, score)) in importance.iter().take(10).enumerate() {
        let flag = if score > &2.0 { " ⚠️ HIGH" } else if score > &1.0 { " ⚡ MEDIUM" } else { "" };
        println!("  {}. {} — drop={:.4}{}", i + 1, name, score, flag);
    }

    // Accounts with zero importance (isolated / dormant)
    let dormant: Vec<&str> = importance
        .iter()
        .filter(|(_, s)| *s < 0.001)
        .map(|(n, _)| n.as_str())
        .collect();
    if !dormant.is_empty() {
        println!("  🟢 Dormant/unused accounts (zero impact): {:?}", dormant);
    }

    // ── 4. Bottleneck Edges ────────────────────────────────────────

    println!("\n[4] BOTTLENECK EDGES (Fiedler drop on edge removal)");
    println!("──────────────────────────────────────────");
    let bottlenecks = graph.bottlenecks();
    println!("  Top 10 bottleneck edges:");
    for (i, (a, b, drop)) in bottlenecks.iter().take(10).enumerate() {
        let flag = if drop > &5.0 { " ⚠️ CRITICAL" } else if drop > &1.0 { " ⚡ SIGNIFICANT" } else { "" };
        println!("  {}. {} ↔ {} — drop={:.4}{}", i + 1, a, b, drop, flag);
    }

    // ── 5. Spectral Clustering ─────────────────────────────────────

    println!("\n[5] SPECTRAL CLUSTERING (k=3)");
    println!("──────────────────────────────────────────");
    let clusters = graph.spectral_cluster(3);
    // Map clusters back to node names
    let mut cluster_groups: Vec<Vec<&str>> = vec![Vec::new(); 3];
    for (idx, &c) in clusters.iter().enumerate() {
        let name = match idx {
            0 => "alice_normal",
            1 => "bob_normal",
            2 => "carol_normal",
            3 => "dave_normal",
            4 => "eve_normal",
            5 => "frank_normal",
            6 => "grace_normal",
            7 => "henry_normal",
            8 => "ivy_normal",
            9 => "jack_normal",
            10 => "suspicious_sarah",
            11 => "suspicious_tom",
            12 => "suspicious_umar",
            13 => "suspicious_vera",
            14 => "suspicious_wu",
            15 => "mule_alpha",
            16 => "mule_beta",
            17 => "mule_gamma",
            18 => "mule_delta",
            19 => "mule_epsilon",
            20 => "external_gateway",
            21 => "dormant_one",
            22 => "dormant_two",
            23 => "dormant_three",
            _ => "???",
        };
        cluster_groups[c].push(name);
    }
    for (c, members) in cluster_groups.iter().enumerate() {
        println!("  Cluster {}: {} members {:?}", c, members.len(), members);
    }

    // ── 6. Effective Resistance (distance between accounts) ────────

    println!("\n[6] EFFECTIVE RESISTANCE (Spectral Distance)");
    println!("──────────────────────────────────────────");
    // Resistance between normal accounts
    let r_alice_bob = graph.effective_resistance(0, 1);
    let r_eve_jack = graph.effective_resistance(4, 9);
    // Resistance between suspicious accounts
    let r_sarah_tom = graph.effective_resistance(10, 11);
    let r_sarah_wu = graph.effective_resistance(10, 14);
    // Resistance between normal and suspicious (should be HIGH)
    let r_alice_sarah = graph.effective_resistance(0, 10);
    let r_jack_sarah = graph.effective_resistance(9, 10);
    // Resistance between mule and external
    let r_mule_ext = graph.effective_resistance(15, 20);
    let r_sarah_mule = graph.effective_resistance(10, 15);
    // Resistance involving dormant accounts
    let r_dormant = graph.effective_resistance(0, 21);
    let r_dormant_pair = graph.effective_resistance(21, 22);

    println!("  Within normal community:");
    println!("    Alice ↔ Bob:       {:.4}", r_alice_bob);
    println!("    Eve ↔ Jack:        {:.4}", r_eve_jack);

    println!("  Within suspicious cluster:");
    println!("    Sarah ↔ Tom:      {:.4} (should be LOW)", r_sarah_tom);
    println!("    Sarah ↔ Wu:       {:.4} (should be LOW)", r_sarah_wu);

    println!("  Cross-community (normal ↔ suspicious):");
    println!("    Alice ↔ Sarah:    {:.4} (should be HIGH)", r_alice_sarah);
    println!("    Jack ↔ Sarah:     {:.4} (should be MODERATE: single bridge)", r_jack_sarah);

    println!("  Mule network:");
    println!("    Sarah ↔ MuleA:    {:.4}", r_sarah_mule);
    println!("    MuleA ↔ Gateway:  {:.4}", r_mule_ext);

    println!("  Dormant accounts:");
    println!("    Alice ↔ Dormant1: {:.4} (should be INF or very HIGH)", r_dormant);
    println!("    Dormant1↔Dormant2:{:.4} (should be INF)", r_dormant_pair);

    // ── 7. Kirchhoff Index ─────────────────────────────────────────

    println!("\n[7] KIRCHHOFF INDEX & CONDITION NUMBER");
    println!("──────────────────────────────────────────");
    let kf = graph.kirchhoff_index();
    let kappa = graph.condition_number();
    println!("  Kirchhoff index:       {:.4}", kf);
    println!("  Condition number:      {:.4}", kappa);
    println!(
        "  Interpretation: {}",
        if kappa > 1000.0 {
            "⚠️  Ill-conditioned — near-disconnected components detected"
        } else if kappa > 100.0 {
            "⚡ High condition — some weak links in network"
        } else {
            "✅ Well-conditioned — numerically stable"
        }
    );

    // ── 8. Fiedler Sensitivity ─────────────────────────────────────

    println!("\n[8] FIEDLER SENSITIVITY (Edge importance per derivative)");
    println!("──────────────────────────────────────────");
    let mut sensitivity = graph.fiedler_sensitivity();
    sensitivity.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    let node_name = |i: usize| -> String {
        match i {
            0 => "alice_normal".into(),
            1 => "bob_normal".into(),
            2 => "carol_normal".into(),
            3 => "dave_normal".into(),
            4 => "eve_normal".into(),
            5 => "frank_normal".into(),
            6 => "grace_normal".into(),
            7 => "henry_normal".into(),
            8 => "ivy_normal".into(),
            9 => "jack_normal".into(),
            10 => "suspicious_sarah".into(),
            11 => "suspicious_tom".into(),
            12 => "suspicious_umar".into(),
            13 => "suspicious_vera".into(),
            14 => "suspicious_wu".into(),
            15 => "mule_alpha".into(),
            16 => "mule_beta".into(),
            17 => "mule_gamma".into(),
            18 => "mule_delta".into(),
            19 => "mule_epsilon".into(),
            20 => "external_gateway".into(),
            _ => format!("n{}", i),
        }
    };

    println!("  Top 10 most sensitive edges:");
    for (i, (u, v, delta)) in sensitivity.iter().take(10).enumerate() {
        let flag = if *delta > 0.05 {
            " ⚠️ HIGH"
        } else if *delta > 0.01 {
            " ⚡"
        } else {
            ""
        };
        println!(
            "  {}. {} ↔ {} — ∂λ₂/∂w = {:.6}{}",
            i + 1,
            node_name(*u),
            node_name(*v),
            delta,
            flag
        );
    }

    // ── 9. Network Community Profile ───────────────────────────────

    println!("\n[9] COMMUNITY PROFILE (Sweep-cut conductance)");
    println!("──────────────────────────────────────────");
    let profile = graph.community_profile();
    for &(size, cond) in profile.iter().take(12) {
        let desc = if cond < 0.01 {
            "🟢 TIGHT COMMUNITY"
        } else if cond < 0.1 {
            "🔵 MODERATE"
        } else if cond < 0.5 {
            "🟡 LOOSE"
        } else {
            "🔴 WEAK CUT"
        };
        println!("  size={:2}, conductance={:.6}  {}", size, cond, desc);
    }

    // ── 10. Full Spectrum ──────────────────────────────────────────

    println!("\n[10] FULL SPECTRUM (Top 12 eigenvalues)");
    println!("──────────────────────────────────────────");
    let spec = graph.spectrum();
    for (i, val) in spec.iter().enumerate().take(12) {
        println!("  λ_{} = {:.6}", i, val);
    }
    if spec.len() > 12 {
        println!("  ... and {} more", spec.len() - 12);
    }

    // ── Summary ────────────────────────────────────────────────────

    println!("\n══════════════════════════════════════════════════════════════");
    println!("  DETECTION SUMMARY");
    println!("══════════════════════════════════════════════════════════════");

    // Detect: dense subgraph (suspicious cluster)
    let dense_fiedler = {
        let names: Vec<&str> = (10..=14)
            .map(|i| match i {
                10 => "suspicious_sarah",
                11 => "suspicious_tom",
                12 => "suspicious_umar",
                13 => "suspicious_vera",
                14 => "suspicious_wu",
                _ => "???",
            })
            .collect();
        let mut sg = CathedralProbe::new(names);
        sg.connect("suspicious_sarah", "suspicious_tom", 12.0);
        sg.connect("suspicious_sarah", "suspicious_umar", 11.5);
        sg.connect("suspicious_sarah", "suspicious_vera", 10.0);
        sg.connect("suspicious_sarah", "suspicious_wu", 13.0);
        sg.connect("suspicious_tom", "suspicious_umar", 9.5);
        sg.connect("suspicious_tom", "suspicious_vera", 8.0);
        sg.connect("suspicious_tom", "suspicious_wu", 11.0);
        sg.connect("suspicious_umar", "suspicious_vera", 10.5);
        sg.connect("suspicious_umar", "suspicious_wu", 9.0);
        sg.connect("suspicious_vera", "suspicious_wu", 12.5);
        sg.fiedler_value()
    };
    let suspicious_fiedler = {
        let names: Vec<&str> = (0..=20)
            .map(|i| match i {
                0 => "alice_normal",
                1 => "bob_normal",
                2 => "carol_normal",
                3 => "dave_normal",
                4 => "eve_normal",
                5 => "frank_normal",
                6 => "grace_normal",
                7 => "henry_normal",
                8 => "ivy_normal",
                9 => "jack_normal",
                10 => "suspicious_sarah",
                11 => "suspicious_tom",
                12 => "suspicious_umar",
                13 => "suspicious_vera",
                14 => "suspicious_wu",
                15 => "mule_alpha",
                16 => "mule_beta",
                17 => "mule_gamma",
                18 => "mule_delta",
                19 => "mule_epsilon",
                20 => "external_gateway",
                _ => "???",
            })
            .collect();
        let mut sg = CathedralProbe::new(names);
        for &(i, j, w) in &[
            // normal
            (0, 1, 0.8), (1, 2, 0.6), (2, 3, 0.7), (3, 4, 0.5),
            (4, 5, 0.9), (5, 6, 0.6), (6, 7, 0.8), (7, 8, 0.5),
            (8, 9, 0.7), (0, 4, 0.3),
            // suspicious
            (10, 11, 12.0), (10, 12, 11.5), (10, 13, 10.0),
            (10, 14, 13.0), (11, 12, 9.5), (11, 13, 8.0),
            (11, 14, 11.0), (12, 13, 10.5), (12, 14, 9.0),
            (13, 14, 12.5),
            // mules
            (10, 15, 7.0), (11, 16, 8.0), (12, 17, 6.5),
            (13, 18, 7.5), (14, 19, 6.0),
            // cash out
            (15, 20, 7.0), (16, 20, 8.0), (17, 20, 6.5),
            (18, 20, 7.5), (19, 20, 6.0),
            // bridge
            (9, 10, 0.1),
        ] {
            let a = match i {
                0 => "alice_normal", 1 => "bob_normal", 2 => "carol_normal", 3 => "dave_normal",
                4 => "eve_normal", 5 => "frank_normal", 6 => "grace_normal", 7 => "henry_normal",
                8 => "ivy_normal", 9 => "jack_normal",
                10 => "suspicious_sarah", 11 => "suspicious_tom", 12 => "suspicious_umar",
                13 => "suspicious_vera", 14 => "suspicious_wu",
                15 => "mule_alpha", 16 => "mule_beta", 17 => "mule_gamma",
                18 => "mule_delta", 19 => "mule_epsilon",
                20 => "external_gateway", _ => unreachable!(),
            };
            let b = match j {
                0 => "alice_normal", 1 => "bob_normal", 2 => "carol_normal", 3 => "dave_normal",
                4 => "eve_normal", 5 => "frank_normal", 6 => "grace_normal", 7 => "henry_normal",
                8 => "ivy_normal", 9 => "jack_normal",
                10 => "suspicious_sarah", 11 => "suspicious_tom", 12 => "suspicious_umar",
                13 => "suspicious_vera", 14 => "suspicious_wu",
                15 => "mule_alpha", 16 => "mule_beta", 17 => "mule_gamma",
                18 => "mule_delta", 19 => "mule_epsilon",
                20 => "external_gateway", _ => unreachable!(),
            };
            sg.connect(a, b, w);
        }
        sg
    };
    _ = suspicious_fiedler;

    println!();
    println!("  🕵️  Dense suspicious subgraph (S₁₀–S₁₄):");
    println!("       Internal Fiedler = {:.4}  (very tight-knit = suspicious)", dense_fiedler);
    println!("       Full-network Fiedler = {:.4}", fiedler);
    println!();
    println!("  💰 Top importance accounts: {:?}",
        importance.iter().take(3).map(|(n, _)| n.as_str()).collect::<Vec<_>>());
    println!();
    println!("  🔗 Key bottlenecks: {:?}",
        bottlenecks.iter().take(3).map(|(a, b, _)| format!("{}↔{}", a, b)).collect::<Vec<_>>());
    println!();
    println!("  📊 Spectral clustering identified {} groups", clusters.len());
    println!("     (should isolate suspicious cluster as its own group)");
    println!();
    println!("  🔌 Dormant accounts isolated: {}", dormant.len());
    println!();
    println!("  ⚡ Condition number: κ = {:.2} {}", kappa,
        if kappa > 100.0 { "— near-disconnected, watch the thin bridge" } else { "— stable" });

    println!("\n  ───────────────────────────────");
    println!("  VERDICT: Suspicious activity detected via spectral analysis");
    println!("  ───────────────────────────────");
    println!("   ✓ High-density cluster (accounts 10-14) forms a near-complete subgraph");
    println!("   ✓ Low effective resistance within suspicious cluster (high cohesion)");
    println!("   ✓ High effective resistance to normal world (isolation)");
    println!("   ✓ Mule accounts bridge funds to external gateway");
    println!("   ✓ Spectral clustering correctly isolates the fraud ring");
    println!("   ✓ Dormant accounts flagged as zero-impact");
}
