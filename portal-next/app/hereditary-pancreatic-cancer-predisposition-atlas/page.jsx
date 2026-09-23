"use client";
import { useEffect, useState } from "react";

const GENE_COLORS = {
  BRCA2:  "#e67e22",
  CDKN2A: "#e74c3c",
  ATM:    "#2980b9",
  PALB2:  "#8e44ad",
  STK11:  "#27ae60",
  BRCA1:  "#f39c12",
  MLH1:   "#16a085",
  TP53:   "#c0392b",
};

const GENE_INFO = {
  BRCA2:  { full: "BRCA2 / FANCD1 (HR Mediator, RAD51 Loader)",      locus: "13q12.3",  size: "3418 aa / 384 kDa", inh: "AD LOF" },
  CDKN2A: { full: "CDKN2A / p16-INK4a / p14-ARF (CDK4/6 Inhibitor)", locus: "9p21.3",   size: "156 aa / 17 kDa",   inh: "AD LOF" },
  ATM:    { full: "ATM (PI3K-like DSB Kinase)",                        locus: "11q22.3",  size: "3056 aa / 350 kDa", inh: "AD LOF" },
  PALB2:  { full: "PALB2 / FANCN (BRCA2 Bridge, WD40 Scaffold)",      locus: "16p12.2",  size: "1186 aa / 131 kDa", inh: "AD LOF" },
  STK11:  { full: "STK11 / LKB1 (AMPK Master Kinase, PJS)",           locus: "19p13.3",  size: "433 aa / 48 kDa",   inh: "AD LOF" },
  BRCA1:  { full: "BRCA1 / FANCS (HR Scaffold, RING-BRCT)",           locus: "17q21.31", size: "1863 aa / 208 kDa", inh: "AD LOF" },
  MLH1:   { full: "MLH1 (MMR MutL Homologue 1, Lynch Type 1)",        locus: "3p22.2",   size: "756 aa / 85 kDa",   inh: "AD LOF" },
  TP53:   { full: "TP53 / p53 (Tumour Suppressor, Genome Guardian)",  locus: "17p13.1",  size: "393 aa / 43 kDa",   inh: "AD LOF" },
};

const SLUG = "hereditary-pancreatic-cancer-predisposition-atlas";

export default function HeredPancreaticCancerPredispositionAtlasPage() {
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

  if (loading) return <div className="p-6 text-white">Loading Hereditary Pancreatic Cancer Predisposition Atlas…</div>;
  if (error)   return <div className="p-6 text-red-400">Error: {error}</div>;

  const geneSummary  = overview?.gene_summary  || [];
  const hierarchy    = overview?.pancreatic_risk_hierarchy || {};
  const keyRules     = overview?.key_clinical_rules || [];
  const perGene      = breakdown?.per_gene      || [];
  const brca1v2      = breakdown?.brca1_vs_brca2_comparison || {};
  const hrdVsMsih    = breakdown?.hrd_vs_msi_h_treatment    || {};
  const stk11Rules   = breakdown?.stk11_pjs_key_rules        || {};
  const defGenes     = definitions?.genes                    || [];
  const keyConcepts  = definitions?.key_clinical_concepts    || {};
  const abbreviations = definitions?.abbreviations           || {};

  const TABS = ["overview", "per_gene", "breakdown", "definitions"];

  return (
    <div style={{ background: "#0f1117", minHeight: "100vh", color: "#e0e0e0", fontFamily: "monospace", padding: "24px" }}>
      <div style={{ maxWidth: 1200, margin: "0 auto" }}>

        {/* Header */}
        <div style={{ marginBottom: 24, borderBottom: "2px solid #1a6b4a", paddingBottom: 16 }}>
          <div style={{ fontSize: 11, color: "#7f8c8d", marginBottom: 6 }}>
            🧬 Expert Dashboards → Hereditary Cancer Predisposition Atlases
          </div>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: "#1a6b4a", margin: 0 }}>
            🏥 Hereditary Pancreatic Cancer Predisposition Atlas
          </h1>
          <div style={{ fontSize: 12, color: "#95a5a6", marginTop: 4 }}>
            Complete 8-Gene Reference · BRCA2-CDKN2A-ATM-PALB2-STK11-BRCA1-MLH1-TP53 ·{" "}
            320-Patient Aggregate (8×40, seeds 3478–3485)
          </div>
        </div>

        {/* KPI Cards */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 12, marginBottom: 24 }}>
          {[
            { label: "Total Patients",  value: overview?.total_patients },
            { label: "PDAC Cases",      value: `${overview?.pdac_cases} (${overview?.pdac_rate_pct}%)` },
            { label: "HRD Cases",       value: `${overview?.hrd_cases} (${overview?.hrd_rate_pct}%)` },
            { label: "MSI-H Cases",     value: overview?.msi_h_cases },
            { label: "PJS Polyp Cases", value: overview?.pjs_polyp_cases },
          ].map(kpi => (
            <div key={kpi.label} style={{ background: "#1a1d2e", border: "1px solid #2c3e50", borderRadius: 8, padding: 14, textAlign: "center" }}>
              <div style={{ fontSize: 20, fontWeight: 700, color: "#1a6b4a" }}>{kpi.value}</div>
              <div style={{ fontSize: 11, color: "#95a5a6", marginTop: 4 }}>{kpi.label}</div>
            </div>
          ))}
        </div>

        {/* Pancreatic Risk Hierarchy */}
        <div style={{ background: "#1a1d2e", border: "1px solid #2c3e50", borderRadius: 8, padding: 14, marginBottom: 16 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: "#e67e22", marginBottom: 10 }}>
            🔺 PANCREATIC CANCER RISK HIERARCHY (lifetime risk, highest → lowest)
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 8 }}>
            {Object.entries(hierarchy).map(([gene, risk], i) => (
              <div key={gene} style={{ background: "#0f1117", borderRadius: 6, padding: "8px 10px", borderLeft: `3px solid ${GENE_COLORS[gene.split("_")[0]] || "#2c3e50"}` }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: GENE_COLORS[gene.split("_")[0]] || "#95a5a6" }}>{gene.replace(/_/g, "/")}</div>
                <div style={{ fontSize: 10, color: "#bdc3c7", marginTop: 2 }}>{risk}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Critical Rules Banner */}
        <div style={{ background: "#1a1d2e", border: "1px solid #e67e22", borderRadius: 8, padding: 14, marginBottom: 24 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: "#e67e22", marginBottom: 8 }}>
            ⚠ CRITICAL CLINICAL RULES — DO NOT MISS
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
            {keyRules.slice(0, 8).map((r, i) => (
              <div key={i} style={{ fontSize: 10, color: "#ecf0f1", background: "#0f1117", borderRadius: 4, padding: "5px 8px" }}>
                • {r}
              </div>
            ))}
          </div>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 8, marginBottom: 20, borderBottom: "1px solid #2c3e50", paddingBottom: 8 }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              padding: "6px 16px", borderRadius: 6, border: "none", cursor: "pointer", fontSize: 12,
              background: tab === t ? "#1a6b4a" : "#1a1d2e",
              color:      tab === t ? "#fff"    : "#7f8c8d",
            }}>
              {t === "overview"    && "Overview"}
              {t === "per_gene"    && "Per Gene"}
              {t === "breakdown"   && "Breakdown"}
              {t === "definitions" && "Definitions"}
            </button>
          ))}
        </div>

        {/* ── OVERVIEW TAB ── */}
        {tab === "overview" && (
          <div>
            <h2 style={{ fontSize: 14, color: "#1a6b4a", marginBottom: 12 }}>Gene Cohort Summary</h2>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                <thead>
                  <tr style={{ background: "#1a1d2e", color: "#7f8c8d" }}>
                    {["Gene", "N", "PDAC", "PDAC%", "HRD", "HRD%", "MSI-H", "PJS Polyps", "Mean Age", "Key Risk"].map(h => (
                      <th key={h} style={{ padding: "7px 10px", textAlign: "left", borderBottom: "1px solid #2c3e50", whiteSpace: "nowrap" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {geneSummary.map((gs, i) => (
                    <tr key={gs.gene} style={{ background: i % 2 === 0 ? "#0f1117" : "#1a1d2e" }}>
                      <td style={{ padding: "7px 10px", color: GENE_COLORS[gs.gene], fontWeight: 700 }}>{gs.gene}</td>
                      <td style={{ padding: "7px 10px", color: "#bdc3c7" }}>{gs.n}</td>
                      <td style={{ padding: "7px 10px", color: "#e74c3c" }}>{gs.pdac}</td>
                      <td style={{ padding: "7px 10px", color: "#e74c3c" }}>{gs.pdac_pct}%</td>
                      <td style={{ padding: "7px 10px", color: "#3498db" }}>{gs.hrd}</td>
                      <td style={{ padding: "7px 10px", color: "#3498db" }}>{gs.hrd_pct}%</td>
                      <td style={{ padding: "7px 10px", color: "#16a085" }}>{gs.msi_h}</td>
                      <td style={{ padding: "7px 10px", color: "#27ae60" }}>{gs.crc}</td>
                      <td style={{ padding: "7px 10px", color: "#95a5a6" }}>{gs.mean_age}yr</td>
                      <td style={{ padding: "7px 10px", color: GENE_COLORS[gs.gene], fontSize: 10 }}>{GENE_INFO[gs.gene]?.full?.split("(")[0]?.trim()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── PER GENE TAB ── */}
        {tab === "per_gene" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            {defGenes.map(dg => (
              <div key={dg.gene} style={{ background: "#1a1d2e", border: `1px solid ${GENE_COLORS[dg.gene]}`, borderRadius: 10, padding: 16 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
                  <div>
                    <span style={{ fontSize: 15, fontWeight: 700, color: GENE_COLORS[dg.gene] }}>{dg.gene}</span>
                    <span style={{ fontSize: 11, color: "#95a5a6", marginLeft: 8 }}>{GENE_INFO[dg.gene]?.locus} · {GENE_INFO[dg.gene]?.size}</span>
                  </div>
                  <span style={{ fontSize: 10, background: "#0f1117", color: "#f39c12", padding: "2px 8px", borderRadius: 4 }}>
                    {GENE_INFO[dg.gene]?.inh}
                  </span>
                </div>
                <div style={{ fontSize: 11, color: "#bdc3c7", marginBottom: 8 }}>{dg.full_name}</div>
                <div style={{ marginBottom: 10 }}>
                  <div style={{ fontSize: 10, fontWeight: 700, color: "#1a6b4a", marginBottom: 4 }}>KEY VARIANTS</div>
                  {(dg.variants || []).map((v, i) => (
                    <div key={i} style={{ fontSize: 10, color: "#95a5a6", background: "#0f1117", borderRadius: 4, padding: "4px 8px", marginBottom: 3 }}>
                      <span style={{ color: GENE_COLORS[dg.gene] }}>{v.variant}</span> — {v.phenotype}
                    </div>
                  ))}
                </div>
                <div>
                  <div style={{ fontSize: 10, fontWeight: 700, color: "#e67e22", marginBottom: 4 }}>SURVEILLANCE</div>
                  {(dg.surveillance || []).map((s, i) => (
                    <div key={i} style={{ fontSize: 10, color: "#95a5a6", marginBottom: 2 }}>• {s}</div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* ── BREAKDOWN TAB ── */}
        {tab === "breakdown" && (
          <div>
            {/* Per-gene breakdown cards */}
            <h2 style={{ fontSize: 14, color: "#1a6b4a", marginBottom: 12 }}>Per-Gene Clinical Breakdown</h2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 24 }}>
              {perGene.map(pg => (
                <div key={pg.gene} style={{ background: "#1a1d2e", border: `1px solid ${GENE_COLORS[pg.gene]}55`, borderRadius: 8, padding: 14 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                    <span style={{ fontSize: 13, fontWeight: 700, color: GENE_COLORS[pg.gene] }}>{pg.gene}</span>
                    <span style={{ fontSize: 10, color: "#95a5a6" }}>{pg.syndrome}</span>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6, marginBottom: 8 }}>
                    <div style={{ background: "#0f1117", borderRadius: 4, padding: "4px 8px", textAlign: "center" }}>
                      <div style={{ fontSize: 13, color: "#e74c3c", fontWeight: 700 }}>{pg.pdac_pct}%</div>
                      <div style={{ fontSize: 9, color: "#7f8c8d" }}>PDAC</div>
                    </div>
                    <div style={{ background: "#0f1117", borderRadius: 4, padding: "4px 8px", textAlign: "center" }}>
                      <div style={{ fontSize: 13, color: "#3498db", fontWeight: 700 }}>{pg.hrd_pct}%</div>
                      <div style={{ fontSize: 9, color: "#7f8c8d" }}>HRD</div>
                    </div>
                    <div style={{ background: "#0f1117", borderRadius: 4, padding: "4px 8px", textAlign: "center" }}>
                      <div style={{ fontSize: 13, color: "#95a5a6", fontWeight: 700 }}>{pg.mean_age}yr</div>
                      <div style={{ fontSize: 9, color: "#7f8c8d" }}>Mean Age</div>
                    </div>
                  </div>
                  <div style={{ fontSize: 10, color: "#e74c3c", marginBottom: 4 }}>
                    <strong>Avoid:</strong> {pg.key_avoid}
                  </div>
                  <div style={{ fontSize: 10, color: "#1a6b4a" }}>
                    <strong>Rule:</strong> {pg.key_rule}
                  </div>
                </div>
              ))}
            </div>

            {/* BRCA1 vs BRCA2 */}
            <div style={{ background: "#1a1d2e", border: "1px solid #e67e22", borderRadius: 8, padding: 14, marginBottom: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#e67e22", marginBottom: 10 }}>
                BRCA1 vs BRCA2 — Pancreatic Cancer Comparison
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                {Object.entries(brca1v2).map(([k, v]) => (
                  <div key={k} style={{ background: "#0f1117", borderRadius: 6, padding: "8px 12px" }}>
                    <div style={{ fontSize: 10, color: "#7f8c8d", marginBottom: 2 }}>{k.replace(/_/g, " ").toUpperCase()}</div>
                    <div style={{ fontSize: 11, color: "#ecf0f1" }}>{v}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* HRD vs MSI-H */}
            <div style={{ background: "#1a1d2e", border: "1px solid #3498db", borderRadius: 8, padding: 14, marginBottom: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#3498db", marginBottom: 10 }}>
                HRD vs MSI-H — Treatment Decision Framework
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                {Object.entries(hrdVsMsih).map(([k, v]) => (
                  <div key={k} style={{ background: "#0f1117", borderRadius: 6, padding: "8px 12px" }}>
                    <div style={{ fontSize: 10, color: "#7f8c8d", marginBottom: 2 }}>{k.replace(/_/g, " ").toUpperCase()}</div>
                    <div style={{ fontSize: 11, color: "#ecf0f1" }}>{v}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* STK11 / PJS Rules */}
            <div style={{ background: "#1a1d2e", border: "1px solid #27ae60", borderRadius: 8, padding: 14 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#27ae60", marginBottom: 10 }}>
                STK11 / Peutz-Jeghers Syndrome — Key Rules
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                {Object.entries(stk11Rules).map(([k, v]) => (
                  <div key={k} style={{ background: "#0f1117", borderRadius: 6, padding: "8px 12px" }}>
                    <div style={{ fontSize: 10, color: "#7f8c8d", marginBottom: 2 }}>{k.replace(/_/g, " ").toUpperCase()}</div>
                    <div style={{ fontSize: 11, color: "#ecf0f1" }}>{v}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ── DEFINITIONS TAB ── */}
        {tab === "definitions" && (
          <div>
            <h2 style={{ fontSize: 14, color: "#1a6b4a", marginBottom: 12 }}>Key Clinical Concepts</h2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 24 }}>
              {Object.entries(keyConcepts).map(([k, v]) => (
                <div key={k} style={{ background: "#1a1d2e", border: "1px solid #2c3e50", borderRadius: 8, padding: 14 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: "#1a6b4a", marginBottom: 8 }}>
                    {k.replace(/_/g, " ").toUpperCase()}
                  </div>
                  <div style={{ fontSize: 11, color: "#bdc3c7", lineHeight: 1.6 }}>{v}</div>
                </div>
              ))}
            </div>

            <h2 style={{ fontSize: 14, color: "#1a6b4a", marginBottom: 12 }}>Abbreviations</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 8 }}>
              {Object.entries(abbreviations).map(([k, v]) => (
                <div key={k} style={{ background: "#1a1d2e", border: "1px solid #2c3e50", borderRadius: 6, padding: "8px 12px", display: "flex", gap: 8 }}>
                  <span style={{ fontSize: 11, fontWeight: 700, color: "#1a6b4a", whiteSpace: "nowrap" }}>{k}</span>
                  <span style={{ fontSize: 11, color: "#95a5a6" }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Footer */}
        <div style={{ marginTop: 32, paddingTop: 16, borderTop: "1px solid #2c3e50", fontSize: 10, color: "#4a5568", textAlign: "center" }}>
          Hereditary Pancreatic Cancer Predisposition Atlas · 8-Gene Reference (BRCA2-CDKN2A-ATM-PALB2-STK11-BRCA1-MLH1-TP53) ·
          320 patients (8×40, seeds 3478–3485) · Agenticfinder Expert Dashboards
        </div>

      </div>
    </div>
  );
}
