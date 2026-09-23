"use client";
import { useEffect, useState } from "react";

const GENE_COLORS = {
  RHBDF2: "#e67e22",
  TP53:   "#8e44ad",
  CDH1:   "#2980b9",
  BRCA2:  "#c0392b",
  ATM:    "#16a085",
  MLH1:   "#27ae60",
  MSH2:   "#d35400",
  PALB2:  "#8e44ad",
};

const GENE_INFO = {
  RHBDF2: { full: "iRhom2 — Inactive Rhomboid Pseudoprotease (ADAM17 Activator)", locus: "17q25.1", size: "315 aa / 35 kDa",   inh: "AD GOF", risk: "95% lifetime esophageal SCC" },
  TP53:   { full: "Tumour Protein p53 (Guardian of the Genome)",                   locus: "17p13.1", size: "393 aa / 43 kDa",   inh: "AD LOF", risk: "3-5% esophageal (LFS)" },
  CDH1:   { full: "E-Cadherin (Epithelial Adhesion Molecule)",                     locus: "16q22.1", size: "882 aa / 97 kDa",   inh: "AD LOF", risk: "2-3× GEJ/Barrett's adenocarcinoma" },
  BRCA2:  { full: "Breast Cancer Gene 2 / FANCD1 (HR Mediator)",                  locus: "13q12.3", size: "3418 aa / 384 kDa", inh: "AD LOF", risk: "2-3× esophageal SCC+adenocarcinoma" },
  ATM:    { full: "Ataxia-Telangiectasia Mutated (PI3K-Like DSB Kinase)",          locus: "11q22.3", size: "3056 aa / 350 kDa", inh: "AR/AD",  risk: "Heterozyg: 2-3× esophageal" },
  MLH1:   { full: "MutL Homolog 1 (MMR MutLα Subunit)",                           locus: "3p22.2",  size: "756 aa / 85 kDa",   inh: "AD LOF", risk: "0.4-1% esophageal adenocarcinoma (Lynch)" },
  MSH2:   { full: "MutS Homolog 2 (MMR MutSα/MutSβ Subunit)",                    locus: "2p21",    size: "934 aa / 104 kDa",  inh: "AD LOF", risk: "0.5% esophageal (Lynch/Muir-Torre)" },
  PALB2:  { full: "Partner and Localiser of BRCA2 / FANCN (BRCA1-BRCA2 Bridge)", locus: "16p12.2", size: "1186 aa / 131 kDa", inh: "AD LOF", risk: "1.8× esophageal (emerging evidence)" },
};

const SLUG = "hereditary-esophageal-cancer-predisposition-atlas";

export default function HeredEsophagealCancerPredispositionAtlasPage() {
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

  if (loading) return <div className="p-6 text-white">Loading Hereditary Esophageal Cancer Predisposition Atlas…</div>;
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
            🏥 Hereditary Esophageal Cancer Predisposition Atlas
          </h1>
          <div style={{ fontSize: 12, color: "#95a5a6", marginTop: 6 }}>
            Complete 8-Gene Reference · RHBDF2-TP53-CDH1-BRCA2-ATM-MLH1-MSH2-PALB2
            · 320-Patient Aggregate (seeds 3462–3469)
          </div>
        </div>

        {/* KPI row */}
        {overview && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 12, marginBottom: 24 }}>
            {[
              { label: "Total Patients",    value: overview.total_patients },
              { label: "Cancer Cases",      value: overview.esophageal_cancer_cases },
              { label: "Cancer Rate",       value: `${overview.cancer_rate_pct}%` },
              { label: "Barrett's Cases",   value: overview.barrett_cases },
              { label: "MSI-H Cases",       value: overview.msi_h_cases },
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
          <div style={{ background: "#1a1208", border: "1px solid #78350f", borderRadius: 8, padding: 16, marginBottom: 24 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#f97316", marginBottom: 10 }}>
              ⚠️ Critical Clinical Rules — Esophageal Cancer Predisposition
            </div>
            {overview.key_clinical_rules.map((r, i) => (
              <div key={i} style={{ fontSize: 11, color: "#fed7aa", marginBottom: 6, paddingLeft: 12, borderLeft: "3px solid #f97316" }}>
                {r}
              </div>
            ))}
          </div>
        )}

        {/* Histology note */}
        {overview?.histology_summary && (
          <div style={{ background: "#0d1b2a", border: "1px solid #1e3a5f", borderRadius: 8, padding: 14, marginBottom: 24 }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: "#60a5fa", marginBottom: 8 }}>📋 Histology Rules by Gene</div>
            {Object.entries(overview.histology_summary).map(([k, v]) => (
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
                    {["Gene", "N", "Cancer", "Barrett's", "MSI-H", "Esophageal Risk"].map(h => (
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
                        <td style={{ padding: "8px 10px", color: "#ef4444" }}>{gs.cancer}</td>
                        <td style={{ padding: "8px 10px", color: "#f59e0b" }}>{gs.barrett}</td>
                        <td style={{ padding: "8px 10px", color: "#8b5cf6" }}>{gs.msi_h}</td>
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
                <div style={{ fontSize: 11, color: "#ef4444", marginBottom: 4 }}><b style={{ color: "#7f8c8d" }}>Risk:</b> {info.risk}</div>
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
            {breakdown.per_gene?.map(gd => (
              <div key={gd.gene} style={{ background: "#1a1d2e", borderRadius: 8, padding: 16, marginBottom: 16 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                  <h4 style={{ margin: 0, color: GENE_COLORS[gd.gene] || ACCENT, fontSize: 14 }}>
                    {gd.gene} — {gd.syndrome}
                  </h4>
                  <span style={{ fontSize: 12, color: "#94a3b8" }}>{gd.n} patients · {gd.cancer_n} cancer ({gd.cancer_pct}%)</span>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, fontSize: 11 }}>
                  <div>
                    <div style={{ color: "#7f8c8d", marginBottom: 4 }}>Esophageal Risk</div>
                    <div style={{ color: "#ef4444" }}>{gd.esophageal_risk}</div>
                  </div>
                  <div>
                    <div style={{ color: "#7f8c8d", marginBottom: 4 }}>Avoid / Note</div>
                    <div style={{ color: "#fca5a5" }}>{gd.key_avoid}</div>
                  </div>
                  <div>
                    <div style={{ color: "#7f8c8d", marginBottom: 4 }}>Mandatory Rule</div>
                    <div style={{ color: "#fbbf24" }}>{gd.key_rule}</div>
                  </div>
                </div>
                {gd.histology_distribution && Object.keys(gd.histology_distribution).length > 0 && (
                  <div style={{ marginTop: 10 }}>
                    <div style={{ fontSize: 10, color: "#7f8c8d", marginBottom: 4 }}>Histology Distribution</div>
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                      {Object.entries(gd.histology_distribution).map(([hist, n]) => (
                        <span key={hist} style={{ background: "#2d3748", borderRadius: 4, padding: "2px 8px", fontSize: 10, color: "#e0e0e0" }}>
                          {hist}: {n}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}

            {/* HRD and Lynch summaries */}
            {breakdown.hrd_specific && (
              <div style={{ background: "#1a1d2e", borderRadius: 8, padding: 20, marginBottom: 16 }}>
                <h3 style={{ color: "#60a5fa", marginTop: 0, marginBottom: 12, fontSize: 14 }}>HRD Therapeutic Rules</h3>
                {Object.entries(breakdown.hrd_specific).map(([key, val]) => (
                  <div key={key} style={{ marginBottom: 8 }}>
                    <span style={{ fontSize: 11, color: "#7f8c8d" }}>{key.replace(/_/g, " ")}: </span>
                    <span style={{ fontSize: 11, color: "#93c5fd" }}>{val}</span>
                  </div>
                ))}
              </div>
            )}
            {breakdown.lynch_msi_summary && (
              <div style={{ background: "#1a1d2e", borderRadius: 8, padding: 20, marginBottom: 16 }}>
                <h3 style={{ color: "#4ade80", marginTop: 0, marginBottom: 12, fontSize: 14 }}>Lynch / MSI-H Rules</h3>
                {Object.entries(breakdown.lynch_msi_summary).map(([key, val]) => (
                  <div key={key} style={{ marginBottom: 8 }}>
                    <span style={{ fontSize: 11, color: "#7f8c8d" }}>{key.replace(/_/g, " ")}: </span>
                    <span style={{ fontSize: 11, color: "#86efac" }}>{val}</span>
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
                    <div style={{ fontSize: 12, color: "#fbbf24", marginBottom: 4, textTransform: "capitalize" }}>
                      {concept.replace(/_/g, " ")}
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
