"use client";
import { useEffect, useState } from "react";

const GENE_COLORS = {
  VHL:  "#1a6b4a",
  FH:   "#c0392b",
  FLCN: "#2980b9",
  SDHB: "#8e44ad",
  BAP1: "#d35400",
  MET:  "#16a085",
  TSC2: "#f39c12",
  PTEN: "#27ae60",
};

const GENE_INFO = {
  VHL:  { full: "Von Hippel-Lindau Tumour Suppressor (HIF-α E3-Ligase Subunit)",    locus: "3p25.3",   size: "284 aa / 30 kDa",   inh: "AD LOF" },
  FH:   { full: "Fumarate Hydratase (TCA Cycle Fumarate→Malate Enzyme)",             locus: "1q43",     size: "510 aa / 55 kDa",   inh: "AD LOF" },
  FLCN: { full: "Folliculin (mTOR-AMPK Lysosomal Scaffold)",                         locus: "17p11.2",  size: "579 aa / 64 kDa",   inh: "AD LOF" },
  SDHB: { full: "Succinate Dehydrogenase Iron-Sulfur Subunit B (Complex II)",        locus: "1p36.13",  size: "280 aa / 32 kDa",   inh: "AD LOF" },
  BAP1: { full: "BRCA1-Associated Protein-1 (Nuclear Deubiquitylase)",               locus: "3p21.1",   size: "729 aa / 80 kDa",   inh: "AD LOF" },
  MET:  { full: "MET Proto-Oncogene (HGF Receptor Tyrosine Kinase)",                 locus: "7q31.2",   size: "1390 aa / 156 kDa", inh: "AD GOF" },
  TSC2: { full: "Tuberin (mTORC1 RAS-GAP Regulator)",                               locus: "16p13.3",  size: "1807 aa / 200 kDa", inh: "AD LOF" },
  PTEN: { full: "Phosphatase and Tensin Homologue (PI3K-AKT Phosphatase)",           locus: "10q23.31", size: "403 aa / 47 kDa",   inh: "AD LOF" },
};

export default function HereditaryRCCPredispositionAtlasPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [tab, setTab] = useState("overview");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = "/api/hereditary-rcc-predisposition-atlas";
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-6 text-white">Loading Hereditary RCC Predisposition Atlas…</div>;
  if (error)   return <div className="p-6 text-red-400">Error: {error}</div>;

  const genes = overview?.genes || [];

  return (
    <div style={{ background: "#0f1117", minHeight: "100vh", color: "#e0e0e0", fontFamily: "monospace", padding: "24px" }}>
      <div style={{ maxWidth: 1200, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 24, borderBottom: "2px solid #1a6b4a", paddingBottom: 16 }}>
          <div style={{ fontSize: 11, color: "#7f8c8d", marginBottom: 6 }}>
            🧬 Expert Dashboards → Hereditary Cancer Predisposition Atlases
          </div>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: "#1a6b4a", margin: 0 }}>
            🏥 Hereditary RCC Predisposition Atlas
          </h1>
          <div style={{ fontSize: 12, color: "#95a5a6", marginTop: 4 }}>
            Complete 8-Gene Reference · VHL-FH-FLCN-SDHB-BAP1-MET-TSC2-PTEN ·{" "}
            320-Patient Aggregate (8×40, seeds 3358–3365)
          </div>
        </div>

        {/* KPI Cards */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, marginBottom: 24 }}>
          {[
            { label: "Total Patients", value: overview?.total_patients },
            { label: "CR Rate",         value: `${overview?.cr_pct}%` },
            { label: "Mean Age at Dx",  value: `${overview?.mean_age_at_dx} yr` },
            { label: "Radiation Rate",  value: `${overview?.radiation_pct}%` },
          ].map(kpi => (
            <div key={kpi.label} style={{ background: "#1a1d2e", border: "1px solid #2c3e50", borderRadius: 8, padding: 14, textAlign: "center" }}>
              <div style={{ fontSize: 22, fontWeight: 700, color: "#1a6b4a" }}>{kpi.value}</div>
              <div style={{ fontSize: 11, color: "#95a5a6", marginTop: 4 }}>{kpi.label}</div>
            </div>
          ))}
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 8, marginBottom: 20, borderBottom: "1px solid #2c3e50", paddingBottom: 0 }}>
          {["overview", "gene-table", "clinical-atlas", "definitions"].map(t => (
            <button key={t} onClick={() => setTab(t)}
              style={{ padding: "8px 16px", background: tab === t ? "#1a6b4a" : "#1a1d2e", color: tab === t ? "#fff" : "#95a5a6",
                border: "1px solid #2c3e50", borderBottom: tab === t ? "2px solid #1a6b4a" : "none",
                borderRadius: "4px 4px 0 0", cursor: "pointer", fontSize: 12, fontFamily: "monospace" }}>
              {t === "overview" ? "Overview" : t === "gene-table" ? "Gene Table" : t === "clinical-atlas" ? "Clinical Atlas" : "Definitions"}
            </button>
          ))}
        </div>

        {/* Overview Tab */}
        {tab === "overview" && (
          <div>
            <div style={{ marginBottom: 20 }}>
              <h3 style={{ color: "#1a6b4a", fontSize: 14, marginBottom: 12 }}>Key Clinical Rules</h3>
              <div style={{ display: "grid", gap: 8 }}>
                {(overview?.key_rules || []).map((rule, i) => (
                  <div key={i} style={{ background: "#1a1d2e", border: "1px solid #2c3e50", borderRadius: 6, padding: 10, fontSize: 12, color: "#e0e0e0" }}>
                    <span style={{ color: "#e74c3c", fontWeight: 700, marginRight: 8 }}>⚠</span>{rule}
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h3 style={{ color: "#1a6b4a", fontSize: 14, marginBottom: 12 }}>Gene Summary (40 patients each)</h3>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10 }}>
                {genes.map(g => {
                  const gs = overview?.gene_summary?.[g];
                  return (
                    <div key={g} style={{ background: "#1a1d2e", border: `1px solid ${GENE_COLORS[g] || "#2c3e50"}`, borderRadius: 8, padding: 12 }}>
                      <div style={{ fontSize: 16, fontWeight: 700, color: GENE_COLORS[g] || "#fff" }}>{g}</div>
                      <div style={{ fontSize: 10, color: "#7f8c8d", marginBottom: 8 }}>{GENE_INFO[g]?.locus} · {GENE_INFO[g]?.inh}</div>
                      <div style={{ fontSize: 11, color: "#95a5a6" }}>n={gs?.n} · CR={gs?.cr_pct}%</div>
                      <div style={{ fontSize: 11, color: "#95a5a6" }}>Mean age: {gs?.mean_age} yr</div>
                      <div style={{ fontSize: 11, color: "#95a5a6" }}>Relapse: {gs?.relapse_pct}%</div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* Gene Table Tab */}
        {tab === "gene-table" && (
          <div>
            <h3 style={{ color: "#1a6b4a", fontSize: 14, marginBottom: 12 }}>Complete Gene Reference Table</h3>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                <thead>
                  <tr style={{ background: "#1a1d2e", borderBottom: "2px solid #1a6b4a" }}>
                    {["Gene", "Locus", "Size", "Inh", "Syndrome", "RCC Risk", "Key Rule"].map(h => (
                      <th key={h} style={{ padding: "8px 10px", textAlign: "left", color: "#1a6b4a", whiteSpace: "nowrap" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {genes.map((g, i) => {
                    const bd = breakdown?.breakdown?.[g];
                    return (
                      <tr key={g} style={{ background: i % 2 === 0 ? "#0f1117" : "#141720", borderBottom: "1px solid #2c3e50" }}>
                        <td style={{ padding: "8px 10px", color: GENE_COLORS[g], fontWeight: 700 }}>{g}</td>
                        <td style={{ padding: "8px 10px", color: "#95a5a6" }}>{GENE_INFO[g]?.locus}</td>
                        <td style={{ padding: "8px 10px", color: "#95a5a6", whiteSpace: "nowrap" }}>{GENE_INFO[g]?.size}</td>
                        <td style={{ padding: "8px 10px", color: g === "MET" ? "#e74c3c" : "#27ae60", fontWeight: 700 }}>{GENE_INFO[g]?.inh}</td>
                        <td style={{ padding: "8px 10px", color: "#e0e0e0", maxWidth: 180 }}>{bd?.syndrome}</td>
                        <td style={{ padding: "8px 10px", color: "#bdc3c7", maxWidth: 200, fontSize: 10 }}>{bd?.rcc_risk?.slice(0, 80)}…</td>
                        <td style={{ padding: "8px 10px", color: "#e74c3c", maxWidth: 200, fontSize: 10, fontWeight: 600 }}>{bd?.key_rule?.slice(0, 80)}…</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Clinical Atlas Tab */}
        {tab === "clinical-atlas" && (
          <div style={{ display: "grid", gap: 16 }}>
            {genes.map(g => {
              const bd = breakdown?.breakdown?.[g];
              return (
                <div key={g} style={{ background: "#1a1d2e", border: `1px solid ${GENE_COLORS[g] || "#2c3e50"}`, borderRadius: 8, padding: 16 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
                    <div>
                      <span style={{ fontSize: 18, fontWeight: 700, color: GENE_COLORS[g] }}>{g}</span>
                      <span style={{ marginLeft: 10, fontSize: 11, color: "#7f8c8d" }}>{GENE_INFO[g]?.full}</span>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <div style={{ fontSize: 11, color: "#95a5a6" }}>{GENE_INFO[g]?.locus} · {GENE_INFO[g]?.size}</div>
                      <div style={{ fontSize: 11, color: g === "MET" ? "#e74c3c" : "#27ae60", fontWeight: 700 }}>{GENE_INFO[g]?.inh}</div>
                    </div>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 10 }}>
                    <div style={{ background: "#0f1117", borderRadius: 6, padding: 10, fontSize: 11 }}>
                      <div style={{ color: "#1a6b4a", fontWeight: 700, marginBottom: 4 }}>RCC Risk</div>
                      <div style={{ color: "#e0e0e0" }}>{bd?.rcc_risk}</div>
                    </div>
                    <div style={{ background: "#0f1117", borderRadius: 6, padding: 10, fontSize: 11 }}>
                      <div style={{ color: "#8e44ad", fontWeight: 700, marginBottom: 4 }}>Pathognomonic</div>
                      <div style={{ color: "#e0e0e0" }}>{bd?.pathognomonic}</div>
                    </div>
                    <div style={{ background: "#0f1117", borderRadius: 6, padding: 10, fontSize: 11, border: "1px solid #c0392b" }}>
                      <div style={{ color: "#e74c3c", fontWeight: 700, marginBottom: 4 }}>⚠ Key Avoid</div>
                      <div style={{ color: "#e0e0e0" }}>{bd?.key_avoid}</div>
                    </div>
                    <div style={{ background: "#0f1117", borderRadius: 6, padding: 10, fontSize: 11 }}>
                      <div style={{ color: "#f39c12", fontWeight: 700, marginBottom: 4 }}>Targeted Rx</div>
                      <div style={{ color: "#e0e0e0" }}>{bd?.targeted_rx}</div>
                    </div>
                  </div>
                  <div style={{ background: "#0f1117", borderRadius: 6, padding: 10, fontSize: 11, marginBottom: 8 }}>
                    <div style={{ color: "#2980b9", fontWeight: 700, marginBottom: 4 }}>Surveillance</div>
                    <div style={{ color: "#bdc3c7" }}>{bd?.surveillance}</div>
                  </div>
                  <div style={{ background: "#c0392b22", border: "1px solid #c0392b44", borderRadius: 6, padding: 10, fontSize: 11 }}>
                    <span style={{ color: "#e74c3c", fontWeight: 700 }}>KEY RULE: </span>
                    <span style={{ color: "#e0e0e0" }}>{bd?.key_rule}</span>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 8, marginTop: 10 }}>
                    {[
                      { label: "Patients", value: bd?.n_patients },
                      { label: "CR%", value: `${bd?.cr_pct}%` },
                      { label: "Mean Age", value: `${bd?.mean_age} yr` },
                      { label: "Relapse%", value: `${bd?.relapse_pct}%` },
                    ].map(m => (
                      <div key={m.label} style={{ background: "#141720", borderRadius: 4, padding: "6px 10px", textAlign: "center" }}>
                        <div style={{ fontSize: 14, fontWeight: 700, color: GENE_COLORS[g] }}>{m.value}</div>
                        <div style={{ fontSize: 10, color: "#7f8c8d" }}>{m.label}</div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Definitions Tab */}
        {tab === "definitions" && (
          <div>
            <h3 style={{ color: "#1a6b4a", fontSize: 14, marginBottom: 12 }}>Clinical Definitions & Key Rules</h3>
            <div style={{ display: "grid", gap: 12, marginBottom: 20 }}>
              {Object.entries(definitions?.definitions || {}).map(([k, v]) => (
                <div key={k} style={{ background: "#1a1d2e", border: "1px solid #2c3e50", borderRadius: 6, padding: 12 }}>
                  <div style={{ color: "#1a6b4a", fontWeight: 700, fontSize: 12, marginBottom: 6 }}>
                    {k.replace(/_/g, " ").toUpperCase()}
                  </div>
                  <div style={{ color: "#bdc3c7", fontSize: 11, lineHeight: 1.6 }}>{v}</div>
                </div>
              ))}
            </div>
            <h3 style={{ color: "#e74c3c", fontSize: 14, marginBottom: 12 }}>Critical Clinical Rules</h3>
            <div style={{ display: "grid", gap: 12 }}>
              {(definitions?.key_clinical_rules || []).map((r, i) => (
                <div key={i} style={{ background: "#1a1d2e", border: "1px solid #c0392b44", borderRadius: 6, padding: 14 }}>
                  <div style={{ color: "#e74c3c", fontWeight: 700, fontSize: 12, marginBottom: 6 }}>
                    ⚠ [{r.gene}] {r.rule}
                  </div>
                  <div style={{ color: "#bdc3c7", fontSize: 11, marginBottom: 6 }}><strong style={{ color: "#f39c12" }}>Rationale:</strong> {r.rationale}</div>
                  <div style={{ color: "#e67e22", fontSize: 11 }}><strong>Consequence:</strong> {r.consequence}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
