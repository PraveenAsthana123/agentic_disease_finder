"use client";
import { useEffect, useState } from "react";

const GENE_COLORS = {
  PTCH1: "#e67e22",
  MSH2:  "#27ae60",
  XPC:   "#e74c3c",
  ERCC2: "#c0392b",
  CYLD:  "#8e44ad",
  TP53:  "#2980b9",
  SUFU:  "#16a085",
  PTCH2: "#d35400",
};

const GENE_INFO = {
  PTCH1: { full: "Patched-1 Receptor (12-TM SHH Inhibitor)", locus: "9q22.32", size: "1447 aa / 161 kDa", inh: "AD LOF", risk: "BCC 100s-1000s lifetime (Gorlin/NBCCS)" },
  MSH2:  { full: "MutS Homolog 2 (MMR MutSα/MutSβ)", locus: "2p21",    size: "935 aa / 105 kDa",   inh: "AD LOF", risk: "Sebaceous carcinoma PATHOGNOMONIC (Muir-Torre)" },
  XPC:   { full: "Xeroderma Pigmentosum Group C (GGR-NER Sensor)", locus: "3p25.1",  size: "940 aa / 106 kDa",  inh: "AR LOF", risk: "SCC/BCC 10,000× (XP-C; NO neurodegeneration)" },
  ERCC2: { full: "ERCC2/XPD (TFIIH Helicase Subunit)", locus: "19q13.32", size: "760 aa / 89 kDa",   inh: "AR LOF", risk: "SCC/BCC 10,000× + neurodegeneration (XP-D)" },
  CYLD:  { full: "Cylindromatosis Tumour Suppressor (K63-DUB)", locus: "16q12.1", size: "956 aa / 109 kDa", inh: "AD LOF", risk: "Cylindromas PATHOGNOMONIC (Brooke-Spiegler)" },
  TP53:  { full: "Tumour Protein p53 (Guardian of the Genome)", locus: "17p13.1", size: "393 aa / 43 kDa",   inh: "AD LOF", risk: "SCC/BCC 5-10× LFS; AVOID RADIATION ABSOLUTELY" },
  SUFU:  { full: "Suppressor of Fused (GLI Sequestration)", locus: "10q24.32", size: "484 aa / 54 kDa",  inh: "AD LOF", risk: "Adult BCC 5-10× + meningioma 3-5×" },
  PTCH2: { full: "Patched-2 Receptor (Gorlin Variant)", locus: "1p32.3",  size: "1203 aa / 137 kDa", inh: "AD LOF", risk: "BCC 5-20× Gorlin variant (less severe than PTCH1)" },
};

const SLUG = "hereditary-cutaneous-malignancy-predisposition-atlas";

export default function HeredCutaneousMalignancyPredispositionAtlasPage() {
  const [overview, setOverview]       = useState(null);
  const [breakdown, setBreakdown]     = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [tab, setTab]                 = useState("overview");
  const [loading, setLoading]         = useState(true);
  const [error, setError]             = useState(null);

  useEffect(() => {
    const base = `/api/${SLUG}`;
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-6 text-white">Loading Hereditary Cutaneous Malignancy Predisposition Atlas…</div>;
  if (error)   return <div className="p-6 text-red-400">Error: {error}</div>;

  const ACCENT = "#e67e22";

  return (
    <div style={{ background: "#0f1117", minHeight: "100vh", color: "#e0e0e0", fontFamily: "monospace", padding: "24px" }}>
      <div style={{ maxWidth: 1200, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 24, borderBottom: `2px solid ${ACCENT}`, paddingBottom: 16 }}>
          <div style={{ fontSize: 11, color: "#7f8c8d", marginBottom: 6 }}>
            🧬 Expert Dashboards → Hereditary Cancer Predisposition Atlases
          </div>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: ACCENT, margin: 0 }}>
            🏥 Hereditary Cutaneous Malignancy Predisposition Atlas
          </h1>
          <div style={{ fontSize: 12, color: "#95a5a6", marginTop: 6 }}>
            Complete 8-Gene Reference · PTCH1-MSH2-XPC-ERCC2-CYLD-TP53-SUFU-PTCH2
            · 320-Patient Aggregate (seeds 3470–3477)
          </div>
        </div>

        {/* KPI row */}
        {overview && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 12, marginBottom: 24 }}>
            {[
              { label: "Total Patients",      value: overview.total_patients },
              { label: "BCC Cases",           value: `${overview.bcc_cases} (${overview.bcc_rate_pct}%)` },
              { label: "SCC Cases",           value: `${overview.scc_cases} (${overview.scc_rate_pct}%)` },
              { label: "Sebaceous Cases",     value: overview.sebaceous_cases },
              { label: "Cylindroma Cases",    value: overview.cylindroma_cases },
            ].map(k => (
              <div key={k.label} style={{ background: "#1a1d2e", borderRadius: 8, padding: "12px 16px", textAlign: "center" }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: ACCENT }}>{k.value}</div>
                <div style={{ fontSize: 11, color: "#7f8c8d", marginTop: 4 }}>{k.label}</div>
              </div>
            ))}
          </div>
        )}

        {/* Critical rules banner */}
        {overview?.key_clinical_rules && (
          <div style={{ background: "#1a1208", border: "1px solid #78350f", borderRadius: 8, padding: 16, marginBottom: 24 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#f97316", marginBottom: 10 }}>
              ⚠️ Critical Clinical Rules — Cutaneous Malignancy Predisposition
            </div>
            {overview.key_clinical_rules.map((r, i) => (
              <div key={i} style={{ fontSize: 11, color: "#fed7aa", marginBottom: 6, paddingLeft: 12, borderLeft: "3px solid #f97316" }}>
                {r}
              </div>
            ))}
          </div>
        )}

        {/* Histology distinctions */}
        {overview?.histology_distinctions && (
          <div style={{ background: "#0d1b2a", border: "1px solid #1e3a5f", borderRadius: 8, padding: 14, marginBottom: 24 }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: "#60a5fa", marginBottom: 8 }}>📋 Histology Rules by Gene Group</div>
            {Object.entries(overview.histology_distinctions).map(([k, v]) => (
              <div key={k} style={{ fontSize: 11, color: "#93c5fd", marginBottom: 4 }}>
                <b style={{ color: "#fbbf24" }}>{k.replace(/_/g," ")}:</b> {v}
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
            <div style={{ background: "#1a1d2e", borderRadius: 8, padding: 20, marginBottom: 20 }}>
              <h3 style={{ color: ACCENT, marginTop: 0, marginBottom: 16, fontSize: 14 }}>Gene Cohort Summary</h3>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead>
                  <tr style={{ color: "#7f8c8d" }}>
                    {["Gene", "N", "BCC", "SCC", "Malignancy %", "Mean Age", "Cutaneous Risk"].map(h => (
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
                        <td style={{ padding: "8px 10px", color: "#f97316" }}>{gs.bcc}</td>
                        <td style={{ padding: "8px 10px", color: "#ef4444" }}>{gs.scc}</td>
                        <td style={{ padding: "8px 10px", color: "#fbbf24" }}>{gs.malignancy_pct}%</td>
                        <td style={{ padding: "8px 10px", color: "#94a3b8" }}>{gs.mean_age}yr</td>
                        <td style={{ padding: "8px 10px", color: "#94a3b8", fontSize: 11 }}>{info.risk || "—"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
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
                <div style={{ fontSize: 11, color: "#ef4444" }}><b style={{ color: "#7f8c8d" }}>Risk:</b> {info.risk}</div>
                {definitions?.genes?.find(g => g.gene === gene)?.variants && (
                  <div style={{ marginTop: 10 }}>
                    <div style={{ fontSize: 10, color: "#7f8c8d", marginBottom: 4 }}>Key Variants</div>
                    {definitions.genes.find(g => g.gene === gene).variants.map(v => (
                      <div key={v.variant} style={{ fontSize: 10, color: "#94a3b8", padding: "2px 0" }}>
                        <span style={{ color: "#fbbf24" }}>{v.variant}</span> → {v.protein_effect}
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
            {breakdown.per_gene?.map(gd => (
              <div key={gd.gene} style={{ background: "#1a1d2e", borderRadius: 8, padding: 16, marginBottom: 16 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                  <h4 style={{ margin: 0, color: GENE_COLORS[gd.gene] || ACCENT, fontSize: 14 }}>
                    {gd.gene} — {gd.syndrome}
                  </h4>
                  <span style={{ fontSize: 12, color: "#94a3b8" }}>
                    n={gd.n} · BCC {gd.bcc_pct}% · SCC {gd.scc_pct}% · Malignancy {gd.malignancy_pct}% · mean {gd.mean_age}yr
                  </span>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, fontSize: 11 }}>
                  <div><div style={{ color: "#7f8c8d", marginBottom: 4 }}>Key Avoid</div><div style={{ color: "#fca5a5" }}>{gd.key_avoid}</div></div>
                  <div><div style={{ color: "#7f8c8d", marginBottom: 4 }}>Mandatory Rule</div><div style={{ color: "#fbbf24" }}>{gd.key_rule}</div></div>
                  <div>
                    <div style={{ color: "#7f8c8d", marginBottom: 4 }}>Special Findings</div>
                    <div style={{ color: "#86efac", fontSize: 10 }}>
                      {gd.okc_n > 0 && `OKC: ${gd.okc_n} · `}
                      {gd.sebaceous_n > 0 && `Sebaceous: ${gd.sebaceous_n} · `}
                      {gd.cylindroma_n > 0 && `Cylindroma: ${gd.cylindroma_n} · `}
                      {gd.neuro_n > 0 && `Neuro: ${gd.neuro_n}`}
                    </div>
                  </div>
                </div>
              </div>
            ))}

            {/* Comparison panels */}
            {breakdown.xp_comparison && (
              <div style={{ background: "#1a1d2e", borderRadius: 8, padding: 20, marginBottom: 16 }}>
                <h3 style={{ color: "#60a5fa", marginTop: 0, marginBottom: 12, fontSize: 14 }}>XP-C vs XP-D (XPC vs ERCC2)</h3>
                {Object.entries(breakdown.xp_comparison).map(([k, v]) => (
                  <div key={k} style={{ marginBottom: 8 }}>
                    <span style={{ fontSize: 11, color: "#7f8c8d" }}>{k.replace(/_/g, " ")}: </span>
                    <span style={{ fontSize: 11, color: "#93c5fd" }}>{v}</span>
                  </div>
                ))}
              </div>
            )}
            {breakdown.shh_pathway_comparison && (
              <div style={{ background: "#1a1d2e", borderRadius: 8, padding: 20, marginBottom: 16 }}>
                <h3 style={{ color: "#fbbf24", marginTop: 0, marginBottom: 12, fontSize: 14 }}>SHH Pathway Comparison (PTCH1 / PTCH2 / SUFU)</h3>
                {Object.entries(breakdown.shh_pathway_comparison).map(([k, v]) => (
                  <div key={k} style={{ marginBottom: 8 }}>
                    <span style={{ fontSize: 11, color: "#7f8c8d" }}>{k.replace(/_/g, " ")}: </span>
                    <span style={{ fontSize: 11, color: "#fef9c3" }}>{v}</span>
                  </div>
                ))}
              </div>
            )}
            {breakdown.muir_torre_diagnosis && (
              <div style={{ background: "#1a1d2e", borderRadius: 8, padding: 20, marginBottom: 16 }}>
                <h3 style={{ color: "#4ade80", marginTop: 0, marginBottom: 12, fontSize: 14 }}>Muir-Torre Diagnosis Rules (MSH2)</h3>
                {Object.entries(breakdown.muir_torre_diagnosis).map(([k, v]) => (
                  <div key={k} style={{ marginBottom: 8 }}>
                    <span style={{ fontSize: 11, color: "#7f8c8d" }}>{k.replace(/_/g, " ")}: </span>
                    <span style={{ fontSize: 11, color: "#86efac" }}>{v}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Definitions tab */}
        {tab === "definitions" && definitions && (
          <div>
            {definitions.key_clinical_concepts && (
              <div style={{ background: "#1a1d2e", borderRadius: 8, padding: 20, marginBottom: 20 }}>
                <h3 style={{ color: ACCENT, marginTop: 0, marginBottom: 16, fontSize: 14 }}>Key Clinical Concepts</h3>
                {Object.entries(definitions.key_clinical_concepts).map(([concept, text]) => (
                  <div key={concept} style={{ marginBottom: 14 }}>
                    <div style={{ fontSize: 12, color: "#fbbf24", marginBottom: 4 }}>
                      {concept.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}
                    </div>
                    <div style={{ fontSize: 11, color: "#94a3b8", lineHeight: 1.6 }}>{text}</div>
                  </div>
                ))}
              </div>
            )}
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
