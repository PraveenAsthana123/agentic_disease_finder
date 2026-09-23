"use client";
import { useEffect, useState } from "react";

const GENE_COLORS = {
  FANCA:  "#c0392b",
  FANCC:  "#e67e22",
  FANCD2: "#d35400",
  FANCG:  "#e74c3c",
  TP53:   "#8e44ad",
  ATM:    "#2980b9",
  NBN:    "#16a085",
  CDKN2A: "#27ae60",
};

const GENE_INFO = {
  FANCA:  { full: "FA Complementation Group A (Core Complex Scaffold)", locus: "16q24.3", size: "1455 aa / 163 kDa", inh: "AR LOF", risk: "500-700×" },
  FANCC:  { full: "FA Complementation Group C (FANCE Chaperone)", locus: "9q22.32",  size: "558 aa / 63 kDa",   inh: "AR LOF", risk: "500-700×" },
  FANCD2: { full: "FA Complementation Group D2 (Ubiquitination Sensor K561)", locus: "3p25.3",  size: "1471 aa / 163 kDa", inh: "AR LOF", risk: "500-700×" },
  FANCG:  { full: "FA Complementation Group G / XRCC9 (NBS1-Linked Scaffold)", locus: "9p13.3",  size: "622 aa / 70 kDa",   inh: "AR LOF", risk: "500-700×" },
  TP53:   { full: "Tumour Protein p53 (Guardian of the Genome)", locus: "17p13.1", size: "393 aa / 43 kDa",   inh: "AD LOF", risk: "30-35% cumulative" },
  ATM:    { full: "Ataxia Telangiectasia Mutated (PI3K-Like DSB Kinase)", locus: "11q22.3", size: "3056 aa / 350 kDa", inh: "AR/AD", risk: "Heterozyg: 2-5× / Biallelic: ≥10×" },
  NBN:    { full: "Nibrin / NBS1 (MRN Complex DSB Sensor)", locus: "8q21.3",  size: "754 aa / 85 kDa",   inh: "AR/AD", risk: "Heterozyg: 2-3× / Biallelic: NBS" },
  CDKN2A: { full: "Cyclin-Dependent Kinase Inhibitor 2A / p16-INK4a", locus: "9p21.3",  size: "156 aa / 15 kDa",   inh: "AD LOF", risk: "5-10× HPV-neg HNSCC" },
};

export default function HereditaryHeadNeckCancerPredispositionAtlasPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [tab, setTab] = useState("overview");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = "/api/hereditary-head-neck-cancer-predisposition-atlas";
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-6 text-white">Loading Hereditary Head &amp; Neck Cancer Predisposition Atlas…</div>;
  if (error)   return <div className="p-6 text-red-400">Error: {error}</div>;

  const ACCENT = "#c0392b";

  return (
    <div style={{ background: "#0f1117", minHeight: "100vh", color: "#e0e0e0", fontFamily: "monospace", padding: "24px" }}>
      <div style={{ maxWidth: 1200, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 24, borderBottom: `2px solid ${ACCENT}`, paddingBottom: 16 }}>
          <div style={{ fontSize: 11, color: "#7f8c8d", marginBottom: 6 }}>
            🧬 Expert Dashboards → Hereditary Cancer Predisposition Atlases
          </div>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: ACCENT, margin: 0 }}>
            🏥 Hereditary Head &amp; Neck Cancer Predisposition Atlas
          </h1>
          <div style={{ fontSize: 12, color: "#95a5a6", marginTop: 6 }}>
            Complete 8-Gene Reference · FANCA-FANCC-FANCD2-FANCG-TP53-ATM-NBN-CDKN2A
            · 320-Patient Aggregate (seeds 3454–3461)
          </div>
        </div>

        {/* KPI row */}
        {overview && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 12, marginBottom: 24 }}>
            {[
              { label: "Total Patients", value: overview.total_patients },
              { label: "HNSCC Cases", value: overview.hnscc_cases },
              { label: "HNSCC Rate", value: `${overview.hnscc_rate_pct}%` },
              { label: "BMF Cases", value: overview.bone_marrow_failure_cases },
              { label: "Alkylating CI", value: overview.alkylating_agent_ci_patients },
            ].map(k => (
              <div key={k.label} style={{ background: "#1a1d2e", borderRadius: 8, padding: "12px 16px", textAlign: "center" }}>
                <div style={{ fontSize: 22, fontWeight: 700, color: ACCENT }}>{k.value}</div>
                <div style={{ fontSize: 11, color: "#7f8c8d", marginTop: 4 }}>{k.label}</div>
              </div>
            ))}
          </div>
        )}

        {/* Critical rules banner */}
        {overview?.key_clinical_rules && (
          <div style={{ background: "#2c1a1a", border: "1px solid #7f1d1d", borderRadius: 8, padding: 16, marginBottom: 24 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#ef4444", marginBottom: 10 }}>
              ⚠️ Critical Clinical Rules — HNSCC Predisposition
            </div>
            {overview.key_clinical_rules.map((r, i) => (
              <div key={i} style={{ fontSize: 11, color: "#fca5a5", marginBottom: 6, paddingLeft: 12, borderLeft: "3px solid #ef4444" }}>
                {r}
              </div>
            ))}
          </div>
        )}

        {/* Tabs */}
        <div style={{ display: "flex", gap: 8, marginBottom: 20 }}>
          {["overview", "per_gene", "breakdown", "definitions"].map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              style={{
                padding: "8px 16px", borderRadius: 6, fontSize: 12, fontWeight: 600, cursor: "pointer",
                background: tab === t ? ACCENT : "#1a1d2e",
                color: tab === t ? "#fff" : "#aaa",
                border: `1px solid ${tab === t ? ACCENT : "#2d3748"}`,
              }}
            >
              {t === "overview" ? "Overview" : t === "per_gene" ? "Per Gene" : t === "breakdown" ? "Breakdown" : "Definitions"}
            </button>
          ))}
        </div>

        {/* Overview tab */}
        {tab === "overview" && overview && (
          <div>
            {/* Gene summary table */}
            <div style={{ background: "#1a1d2e", borderRadius: 8, padding: 20, marginBottom: 20 }}>
              <h3 style={{ color: ACCENT, marginTop: 0, marginBottom: 16, fontSize: 14 }}>Gene Cohort Summary</h3>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead>
                  <tr style={{ color: "#7f8c8d" }}>
                    {["Gene", "N", "HNSCC", "BMF", "AML", "Risk Level", "Key Rule"].map(h => (
                      <th key={h} style={{ textAlign: "left", padding: "6px 10px", borderBottom: "1px solid #2d3748" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {overview.gene_summary.map(gs => {
                    const info = GENE_INFO[gs.gene] || {};
                    return (
                      <tr key={gs.gene} style={{ borderBottom: "1px solid #1e2a3a" }}>
                        <td style={{ padding: "8px 10px", color: GENE_COLORS[gs.gene] || ACCENT, fontWeight: 700 }}>{gs.gene}</td>
                        <td style={{ padding: "8px 10px", color: "#e0e0e0" }}>{gs.n}</td>
                        <td style={{ padding: "8px 10px", color: "#ef4444" }}>{gs.hnscc}</td>
                        <td style={{ padding: "8px 10px", color: "#f59e0b" }}>{gs.bmf}</td>
                        <td style={{ padding: "8px 10px", color: "#8b5cf6" }}>{gs.aml}</td>
                        <td style={{ padding: "8px 10px", color: "#94a3b8", fontSize: 11 }}>{info.risk || "—"}</td>
                        <td style={{ padding: "8px 10px", color: "#94a3b8", fontSize: 10 }}>
                          {gs.gene === "FANCA" || gs.gene === "FANCC" || gs.gene === "FANCD2" || gs.gene === "FANCG"
                            ? "No alkylating agents; cetuximab+surgery"
                            : gs.gene === "TP53" ? "Avoid RT absolutely; WB-MRI annually"
                            : gs.gene === "ATM" ? "Reduce RT 20-30%; A-T = IgA def"
                            : gs.gene === "NBN" ? "657del5 Slavic; radiation sensitivity biallelic"
                            : "Annual oral exam + derm age 18yr; pancreatic MRI 40yr"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* FA summary */}
            {overview.fanconi_anemia_summary && (
              <div style={{ background: "#1a1d2e", borderRadius: 8, padding: 20, marginBottom: 20 }}>
                <h3 style={{ color: "#e67e22", marginTop: 0, marginBottom: 12, fontSize: 14 }}>
                  🔬 Fanconi Anaemia — HNSCC Overview
                </h3>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                  <div>
                    <div style={{ fontSize: 11, color: "#7f8c8d", marginBottom: 6 }}>Complementation Groups</div>
                    {overview.fanconi_anemia_summary.complementation_groups.map(g => (
                      <div key={g} style={{ fontSize: 12, color: "#e0e0e0", padding: "3px 0" }}>• {g}</div>
                    ))}
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: "#7f8c8d", marginBottom: 6 }}>Diagnostic Test</div>
                    <div style={{ fontSize: 12, color: "#fbbf24" }}>{overview.fanconi_anemia_summary.deb_mmc_test}</div>
                    <div style={{ marginTop: 10, fontSize: 11, color: "#7f8c8d" }}>Treatment Protocol</div>
                    <div style={{ fontSize: 12, color: "#f87171" }}>{overview.fanconi_anemia_summary.treatment_rule}</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Per Gene tab */}
        {tab === "per_gene" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            {Object.entries(GENE_INFO).map(([gene, info]) => (
              <div key={gene} style={{ background: "#1a1d2e", borderRadius: 8, padding: 16, borderLeft: `4px solid ${GENE_COLORS[gene] || ACCENT}` }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
                  <div>
                    <span style={{ fontSize: 16, fontWeight: 700, color: GENE_COLORS[gene] || ACCENT }}>{gene}</span>
                    <span style={{ fontSize: 10, color: "#7f8c8d", marginLeft: 8 }}>{info.locus}</span>
                  </div>
                  <span style={{ fontSize: 10, background: "#2d3748", color: "#94a3b8", padding: "2px 8px", borderRadius: 4 }}>{info.inh}</span>
                </div>
                <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 8 }}>{info.full}</div>
                <div style={{ fontSize: 11, color: "#e0e0e0", marginBottom: 4 }}><b style={{ color: "#7f8c8d" }}>Size:</b> {info.size}</div>
                <div style={{ fontSize: 11, color: "#ef4444", marginBottom: 4 }}><b style={{ color: "#7f8c8d" }}>HNSCC Risk:</b> {info.risk}</div>
                {/* Variants */}
                {definitions?.genes?.find(g => g.gene === gene)?.variants && (
                  <div style={{ marginTop: 10 }}>
                    <div style={{ fontSize: 10, color: "#7f8c8d", marginBottom: 4 }}>Key Variants</div>
                    {definitions.genes.find(g => g.gene === gene).variants.map(v => (
                      <div key={v.variant} style={{ fontSize: 10, color: "#94a3b8", padding: "2px 0" }}>
                        <span style={{ color: "#fbbf24" }}>{v.variant}</span> → {v.protein_effect} ({v.location})
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Breakdown tab */}
        {tab === "breakdown" && breakdown && (
          <div>
            {/* Per-gene detailed breakdown */}
            {breakdown.per_gene?.map(gd => (
              <div key={gd.gene} style={{ background: "#1a1d2e", borderRadius: 8, padding: 16, marginBottom: 16 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                  <h4 style={{ margin: 0, color: GENE_COLORS[gd.gene] || ACCENT, fontSize: 14 }}>
                    {gd.gene} — {gd.syndrome}
                  </h4>
                  <span style={{ fontSize: 12, color: "#94a3b8" }}>{gd.n} patients · {gd.hnscc_n} HNSCC ({gd.hnscc_pct}%)</span>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, fontSize: 11 }}>
                  <div>
                    <div style={{ color: "#7f8c8d", marginBottom: 4 }}>HNSCC Risk</div>
                    <div style={{ color: "#ef4444" }}>{gd.hnscc_risk}</div>
                  </div>
                  <div>
                    <div style={{ color: "#7f8c8d", marginBottom: 4 }}>Avoid</div>
                    <div style={{ color: "#fca5a5" }}>{gd.key_avoid}</div>
                  </div>
                  <div>
                    <div style={{ color: "#7f8c8d", marginBottom: 4 }}>Mandatory Rule</div>
                    <div style={{ color: "#fbbf24" }}>{gd.key_rule}</div>
                  </div>
                </div>
                {gd.hnscc_site_distribution && Object.keys(gd.hnscc_site_distribution).length > 0 && (
                  <div style={{ marginTop: 10 }}>
                    <div style={{ fontSize: 10, color: "#7f8c8d", marginBottom: 4 }}>HNSCC Site Distribution</div>
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                      {Object.entries(gd.hnscc_site_distribution).map(([site, n]) => (
                        <span key={site} style={{ background: "#2d3748", borderRadius: 4, padding: "2px 8px", fontSize: 10, color: "#e0e0e0" }}>
                          {site}: {n}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}

            {/* DDR pathway summary */}
            {breakdown.ddr_pathway_summary && (
              <div style={{ background: "#1a1d2e", borderRadius: 8, padding: 20, marginBottom: 16 }}>
                <h3 style={{ color: "#60a5fa", marginTop: 0, marginBottom: 12, fontSize: 14 }}>DDR Pathway Summary</h3>
                {Object.entries(breakdown.ddr_pathway_summary).map(([key, val]) => (
                  <div key={key} style={{ marginBottom: 8 }}>
                    <span style={{ fontSize: 11, color: "#7f8c8d" }}>{key.replace(/_/g, " ")}: </span>
                    <span style={{ fontSize: 11, color: "#94a3b8" }}>{val}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Definitions tab */}
        {tab === "definitions" && definitions && (
          <div>
            {/* Key clinical concepts */}
            {definitions.key_clinical_concepts && (
              <div style={{ background: "#1a1d2e", borderRadius: 8, padding: 20, marginBottom: 20 }}>
                <h3 style={{ color: ACCENT, marginTop: 0, marginBottom: 16, fontSize: 14 }}>Key Clinical Concepts</h3>
                {Object.entries(definitions.key_clinical_concepts).map(([concept, text]) => (
                  <div key={concept} style={{ marginBottom: 14 }}>
                    <div style={{ fontSize: 12, color: "#fbbf24", marginBottom: 4, textTransform: "capitalize" }}>
                      {concept.replace(/_/g, " ")}
                    </div>
                    <div style={{ fontSize: 11, color: "#94a3b8", lineHeight: 1.6 }}>{text}</div>
                  </div>
                ))}
              </div>
            )}

            {/* Abbreviations */}
            {definitions.abbreviations && (
              <div style={{ background: "#1a1d2e", borderRadius: 8, padding: 20, marginBottom: 20 }}>
                <h3 style={{ color: "#60a5fa", marginTop: 0, marginBottom: 12, fontSize: 14 }}>Abbreviations</h3>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                  {Object.entries(definitions.abbreviations).map(([abbr, full]) => (
                    <div key={abbr} style={{ fontSize: 11 }}>
                      <span style={{ color: "#fbbf24", fontWeight: 700 }}>{abbr}</span>
                      <span style={{ color: "#7f8c8d" }}> — </span>
                      <span style={{ color: "#94a3b8" }}>{full}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
