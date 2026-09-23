"use client";
import { useEffect, useState } from "react";

const GENE_COLORS = {
  MLH1:  "#1a6b4a",
  MSH2:  "#c0392b",
  MSH6:  "#2980b9",
  PMS2:  "#8e44ad",
  PTEN:  "#d35400",
  TP53:  "#16a085",
  BRCA1: "#f39c12",
  STK11: "#27ae60",
};

const GENE_INFO = {
  MLH1:  { full: "MutL Homolog 1 (MutLα Scaffold / Lynch Type 1)", locus: "3p22.2",   size: "756 aa / 85 kDa",   inh: "AD LOF", risk: "40-60% endometrial" },
  MSH2:  { full: "MutS Homolog 2 (MutSα+MutSβ / Lynch Type 2 / Muir-Torre)", locus: "2p21",     size: "934 aa / 105 kDa",  inh: "AD LOF", risk: "40-60% endometrial" },
  MSH6:  { full: "MutS Homolog 6 (MutSα / Lynch Type 3 / Endometrial-Dominant)", locus: "2p16.3",   size: "1360 aa / 160 kDa", inh: "AD LOF", risk: "71% HIGHEST" },
  PMS2:  { full: "PMS1 Homolog 2 (MutLα Endonuclease / Lynch Type 4 / CMMRD)", locus: "7p22.1",   size: "862 aa / 96 kDa",   inh: "AD LOF / AR CMMRD", risk: "15-26% lowest" },
  PTEN:  { full: "Phosphatase and Tensin Homolog (PI3K-mTOR Phosphatase / Cowden)", locus: "10q23.31", size: "403 aa / 47 kDa",   inh: "AD LOF", risk: "28-44% Cowden" },
  TP53:  { full: "Tumour Protein p53 (Guardian of Genome / Li-Fraumeni)", locus: "17p13.1",  size: "393 aa / 43 kDa",   inh: "AD LOF", risk: "Uterine serous 2-5×" },
  BRCA1: { full: "Breast Cancer Gene 1 (RING-BRCT HR Scaffold / HBOC1)", locus: "17q21.31", size: "1863 aa / 208 kDa", inh: "AD LOF", risk: "Uterine serous 2-3×" },
  STK11: { full: "Serine/Threonine Kinase 11 (LKB1 / AMPK Kinase / PJS)", locus: "19p13.3",  size: "433 aa / 48 kDa",   inh: "AD LOF", risk: "13% endometrial" },
};

export default function HereditaryUterineEndometrialCancerPredispositionAtlasPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [tab, setTab] = useState("overview");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = "/api/hereditary-uterine-endometrial-cancer-predisposition-atlas";
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-6 text-white">Loading Hereditary Uterine/Endometrial Cancer Predisposition Atlas…</div>;
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
            🏥 Hereditary Uterine/Endometrial Cancer Predisposition Atlas
          </h1>
          <div style={{ fontSize: 12, color: "#95a5a6", marginTop: 6 }}>
            Complete 8-Gene Reference · MLH1-MSH2-MSH6-PMS2-PTEN-TP53-BRCA1-STK11 ·
            320-Patient Aggregate (8×40, seeds 3382-3389) · Lynch / Cowden / LFS / HBOC / PJS
          </div>
          <div style={{ marginTop: 8, padding: "6px 12px", background: "#1a1a2e", borderRadius: 4, fontSize: 11, color: "#e74c3c", display: "inline-block" }}>
            ⚠ KEY RULES: MSH6 = 71% endometrial HIGHEST | TP53 = AVOID RT ABSOLUTELY | MLH1 IHC-loss = 90% somatic (confirm germline) | PMS2 = MLPA MANDATORY (4 pseudogenes)
          </div>
        </div>

        {/* Cohort Summary */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 10, marginBottom: 20 }}>
          {[
            { label: "Total Patients", value: overview?.total_patients },
            { label: "Genes", value: (overview?.genes||[]).length },
            { label: "Checkpoint Inhibitor %", value: `${overview?.checkpoint_inhibitor_rate_pct}%` },
            { label: "Targeted Therapy %", value: `${overview?.targeted_therapy_rate_pct}%` },
            { label: "RRHE Rate %", value: `${overview?.rrhe_rate_pct}%` },
            { label: "Mean Age Dx", value: overview?.mean_age_at_dx },
          ].map(s => (
            <div key={s.label} style={{ background: "#1a1a2e", borderRadius: 6, padding: "10px 12px", textAlign: "center" }}>
              <div style={{ fontSize: 18, fontWeight: 700, color: "#1a6b4a" }}>{s.value ?? "—"}</div>
              <div style={{ fontSize: 10, color: "#95a5a6", marginTop: 2 }}>{s.label}</div>
            </div>
          ))}
        </div>

        {/* Stage */}
        <div style={{ marginBottom: 16, padding: "8px 12px", background: "#1a1a2e", borderRadius: 6, fontSize: 11 }}>
          <span style={{ color: "#e74c3c", fontWeight: 700 }}>Stage III/IV: {overview?.stage_iii_iv_pct}% </span>
          <span style={{ color: "#95a5a6", marginLeft: 16 }}>Seed range: {overview?.seed_range}</span>
          <span style={{ color: "#f39c12", marginLeft: 16 }}>TP53 RT note: {overview?.tp53_rt_avoidance_note}</span>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 8, marginBottom: 20, borderBottom: "1px solid #2c2c44" }}>
          {["overview", "gene-table", "clinical-atlas", "definitions"].map(t => (
            <button key={t} onClick={() => setTab(t)}
              style={{
                padding: "8px 16px", border: "none", borderRadius: "4px 4px 0 0",
                background: tab === t ? "#1a6b4a" : "#1a1a2e",
                color: tab === t ? "#fff" : "#95a5a6",
                cursor: "pointer", fontSize: 12, fontFamily: "monospace",
                borderBottom: tab === t ? "2px solid #1a6b4a" : "2px solid transparent",
              }}
            >
              {t === "overview" ? "📊 Overview" : t === "gene-table" ? "🧬 Gene Table" : t === "clinical-atlas" ? "🏥 Clinical Atlas" : "📖 Definitions"}
            </button>
          ))}
        </div>

        {/* Overview Tab */}
        {tab === "overview" && (
          <div>
            <h2 style={{ color: "#1a6b4a", fontSize: 16, marginBottom: 12 }}>Gene Distribution & Key Facts</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 20 }}>
              {genes.map(gene => (
                <div key={gene} style={{ background: "#1a1a2e", borderRadius: 6, padding: "10px 14px", borderLeft: `3px solid ${GENE_COLORS[gene] || "#555"}` }}>
                  <div style={{ fontSize: 14, fontWeight: 700, color: GENE_COLORS[gene] || "#fff" }}>{gene}</div>
                  <div style={{ fontSize: 10, color: "#bdc3c7", marginTop: 2 }}>{GENE_INFO[gene]?.locus} · {GENE_INFO[gene]?.size}</div>
                  <div style={{ fontSize: 10, color: "#f39c12", marginTop: 2 }}>Risk: {GENE_INFO[gene]?.risk}</div>
                  <div style={{ fontSize: 10, color: "#95a5a6", marginTop: 2 }}>n={overview?.gene_counts?.[gene]}</div>
                </div>
              ))}
            </div>
            <div style={{ background: "#1a1a2e", borderRadius: 6, padding: 14 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: "#1a6b4a", marginBottom: 8 }}>Key Clinical Facts</div>
              {(overview?.key_facts || []).map((f, i) => (
                <div key={i} style={{ fontSize: 11, color: "#bdc3c7", marginBottom: 5, paddingLeft: 12, borderLeft: "2px solid #1a6b4a" }}>
                  {f}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Gene Table Tab */}
        {tab === "gene-table" && (
          <div>
            <h2 style={{ color: "#1a6b4a", fontSize: 16, marginBottom: 12 }}>Complete 8-Gene Reference Table</h2>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                <thead>
                  <tr style={{ background: "#1a6b4a", color: "#fff" }}>
                    {["Gene", "Locus", "Size", "Syndrome", "Inheritance", "Endometrial Risk", "Pathognomonic"].map(h => (
                      <th key={h} style={{ padding: "8px 10px", textAlign: "left", whiteSpace: "nowrap" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {genes.map((gene, idx) => {
                    const info = breakdown?.breakdown?.[gene]?.gene_info || {};
                    return (
                      <tr key={gene} style={{ background: idx % 2 === 0 ? "#1a1a2e" : "#0f1117", borderBottom: "1px solid #2c2c44" }}>
                        <td style={{ padding: "8px 10px", fontWeight: 700, color: GENE_COLORS[gene] || "#fff" }}>{gene}</td>
                        <td style={{ padding: "8px 10px", color: "#95a5a6" }}>{info.locus || GENE_INFO[gene]?.locus}</td>
                        <td style={{ padding: "8px 10px", color: "#bdc3c7" }}>{GENE_INFO[gene]?.size}</td>
                        <td style={{ padding: "8px 10px", color: "#bdc3c7", maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis" }}>{info.syndrome}</td>
                        <td style={{ padding: "8px 10px", color: "#95a5a6" }}>{info.inheritance}</td>
                        <td style={{ padding: "8px 10px", color: "#f39c12", fontWeight: 600 }}>{info.endometrial_risk}</td>
                        <td style={{ padding: "8px 10px", color: "#e74c3c", maxWidth: 220, overflow: "hidden", textOverflow: "ellipsis" }}>{info.pathognomonic}</td>
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
          <div>
            <h2 style={{ color: "#1a6b4a", fontSize: 16, marginBottom: 12 }}>Per-Gene Clinical Atlas</h2>
            {genes.map(gene => {
              const bd = breakdown?.breakdown?.[gene];
              if (!bd) return null;
              return (
                <div key={gene} style={{ background: "#1a1a2e", borderRadius: 6, marginBottom: 14, borderLeft: `4px solid ${GENE_COLORS[gene] || "#555"}` }}>
                  <div style={{ padding: "12px 14px", borderBottom: "1px solid #2c2c44" }}>
                    <span style={{ fontSize: 15, fontWeight: 700, color: GENE_COLORS[gene] || "#fff" }}>{gene}</span>
                    <span style={{ fontSize: 11, color: "#95a5a6", marginLeft: 12 }}>{bd.gene_info?.locus} · {GENE_INFO[gene]?.size} · {bd.gene_info?.inheritance}</span>
                    <span style={{ fontSize: 11, color: "#f39c12", marginLeft: 12 }}>n={bd.n} · mean age {bd.mean_age}yr</span>
                  </div>
                  <div style={{ padding: "10px 14px", display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
                    {[
                      { label: "Checkpoint Inh %", val: `${bd.checkpoint_inhibitor_pct}%` },
                      { label: "Targeted Rx %", val: `${bd.targeted_therapy_pct}%` },
                      { label: "RRHE %", val: `${bd.rrhe_pct}%` },
                      { label: "Relapse %", val: `${bd.relapse_pct}%` },
                    ].map(m => (
                      <div key={m.label} style={{ textAlign: "center" }}>
                        <div style={{ fontSize: 16, fontWeight: 700, color: GENE_COLORS[gene] || "#fff" }}>{m.val}</div>
                        <div style={{ fontSize: 10, color: "#7f8c8d" }}>{m.label}</div>
                      </div>
                    ))}
                  </div>
                  <div style={{ padding: "8px 14px", fontSize: 11 }}>
                    <div style={{ color: "#e74c3c", marginBottom: 4 }}><strong>⚠ Avoid:</strong> {bd.gene_info?.key_avoid}</div>
                    <div style={{ color: "#27ae60", marginBottom: 4 }}><strong>✓ Rule:</strong> {bd.gene_info?.key_rule}</div>
                    <div style={{ color: "#3498db", marginBottom: 4 }}><strong>Surveillance:</strong> {bd.gene_info?.surveillance}</div>
                    <div style={{ color: "#f39c12" }}><strong>Treatment:</strong> {bd.gene_info?.targeted_rx}</div>
                  </div>
                  <div style={{ padding: "6px 14px 10px", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                    <div>
                      <div style={{ fontSize: 10, color: "#7f8c8d", marginBottom: 3 }}>Top Tumour Types</div>
                      {(bd.top_tumour_types || []).map(t => (
                        <div key={t.type} style={{ fontSize: 10, color: "#bdc3c7" }}>• {t.type} ({t.count})</div>
                      ))}
                    </div>
                    <div>
                      <div style={{ fontSize: 10, color: "#7f8c8d", marginBottom: 3 }}>Top Variants</div>
                      {(bd.top_variants || []).map(v => (
                        <div key={v.variant} style={{ fontSize: 10, color: "#95a5a6", fontFamily: "monospace" }}>• {v.variant} ({v.count})</div>
                      ))}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Definitions Tab */}
        {tab === "definitions" && definitions && (
          <div>
            <h2 style={{ color: "#1a6b4a", fontSize: 16, marginBottom: 12 }}>Clinical Definitions & Key Distinctions</h2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 16 }}>
              {Object.entries(definitions.definitions || {}).map(([key, text]) => (
                <div key={key} style={{ background: "#1a1a2e", borderRadius: 6, padding: 12, borderLeft: "3px solid #1a6b4a" }}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: "#1a6b4a", marginBottom: 6, textTransform: "uppercase" }}>
                    {key.replace(/_/g, " ")}
                  </div>
                  <div style={{ fontSize: 10, color: "#bdc3c7", lineHeight: 1.5 }}>{text}</div>
                </div>
              ))}
            </div>
            <div style={{ background: "#1a1a2e", borderRadius: 6, padding: 14 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: "#e74c3c", marginBottom: 8 }}>⚠ Key Clinical Distinctions</div>
              {(definitions.key_clinical_distinctions || []).map((d, i) => (
                <div key={i} style={{ fontSize: 11, color: "#bdc3c7", marginBottom: 5, paddingLeft: 12, borderLeft: "2px solid #e74c3c" }}>
                  {d}
                </div>
              ))}
            </div>
          </div>
        )}

        <div style={{ marginTop: 24, fontSize: 10, color: "#4a4a6a", textAlign: "center" }}>
          Hereditary-Uterine-Endometrial-Cancer-Predisposition-Atlas · 320 patients (8×40, seeds 3382-3389) ·
          MLH1-MSH2-MSH6-PMS2-PTEN-TP53-BRCA1-STK11 · AgenticFinder Portal
        </div>
      </div>
    </div>
  );
}
