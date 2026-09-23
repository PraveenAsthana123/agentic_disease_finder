"use client";
import { useEffect, useState } from "react";

const GENE_COLORS = {
  TP53:    "#e74c3c",
  CTNNB1:  "#8e44ad",
  CDKN2A:  "#2980b9",
  NF1:     "#27ae60",
  MEN1:    "#d35400",
  PRKAR1A: "#c0392b",
  ARMC5:   "#16a085",
  MAX:     "#f39c12",
};

const GENE_INFO = {
  TP53:    { full: "Tumour protein p53",                                      locus: "17p13.1",  size: "393 aa / 43 kDa",    inh: "AD LOF" },
  CTNNB1:  { full: "β-catenin (CTNNB1)",                                      locus: "3p22.1",   size: "781 aa / 85 kDa",    inh: "AD GOF" },
  CDKN2A:  { full: "Cyclin-dependent kinase inhibitor 2A (p16/p14ARF)",       locus: "9p21.3",   size: "156 aa / 17 kDa",    inh: "AD LOF" },
  NF1:     { full: "Neurofibromin (RAS-GAP)",                                 locus: "17q11.2",  size: "2839 aa / 319 kDa",  inh: "AD LOF" },
  MEN1:    { full: "Menin (MEN1 scaffold protein)",                           locus: "11q13.1",  size: "610 aa / 68 kDa",    inh: "AD LOF" },
  PRKAR1A: { full: "PKA regulatory subunit I alpha (R1α)",                    locus: "17q24.2",  size: "381 aa / 43 kDa",    inh: "AD LOF" },
  ARMC5:   { full: "Armadillo repeat containing 5 (ARMC5)",                   locus: "2p13.3",   size: "1059 aa / 120 kDa",  inh: "AD LOF" },
  MAX:     { full: "MYC-associated factor X (MAX)",                           locus: "14q23.3",  size: "236 aa / 18 kDa",    inh: "AD LOF" },
};

export default function HereditaryACCPredispositionAtlasPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [tab, setTab] = useState("overview");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = "/api/hereditary-acc-predisposition-atlas";
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-6 text-white">Loading Hereditary Adrenocortical Carcinoma Predisposition Atlas…</div>;
  if (error)   return <div className="p-6 text-red-400">Error: {error}</div>;

  const genes = overview?.genes || [];

  return (
    <div style={{ background: "#0f1117", minHeight: "100vh", color: "#e0e0e0", fontFamily: "monospace", padding: "24px" }}>
      <div style={{ maxWidth: 1200, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 24, borderBottom: "2px solid #e74c3c", paddingBottom: 16 }}>
          <div style={{ fontSize: 11, color: "#7f8c8d", marginBottom: 6 }}>
            🧬 Expert Dashboards → Hereditary Cancer Predisposition Atlases
          </div>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: "#e74c3c", margin: 0 }}>
            🏥 Hereditary Adrenocortical Carcinoma Predisposition Atlas
          </h1>
          <div style={{ fontSize: 12, color: "#95a5a6", marginTop: 4 }}>
            Complete 8-Gene Reference · TP53-CTNNB1-CDKN2A-NF1-MEN1-PRKAR1A-ARMC5-MAX ·{" "}
            320-Patient Aggregate (8×40, seeds 3334–3341)
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
              <div style={{ fontSize: 22, fontWeight: 700, color: "#e74c3c" }}>{kpi.value}</div>
              <div style={{ fontSize: 11, color: "#95a5a6", marginTop: 4 }}>{kpi.label}</div>
            </div>
          ))}
        </div>

        {/* Key Rules Banner */}
        <div style={{ background: "#1a1d2e", border: "1px solid #e74c3c", borderRadius: 8, padding: 14, marginBottom: 24 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: "#e74c3c", marginBottom: 8 }}>
            ⚠ CRITICAL CLINICAL RULES — DO NOT MISS
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
            {overview?.key_rules?.map((r, i) => (
              <div key={i} style={{ fontSize: 11, color: "#ecf0f1", background: "#0f1117", borderRadius: 4, padding: "6px 10px" }}>
                • {r}
              </div>
            ))}
          </div>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 8, marginBottom: 20, borderBottom: "1px solid #2c3e50", paddingBottom: 8 }}>
          {["overview", "gene-table", "clinical-atlas", "definitions"].map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              style={{
                padding: "6px 16px", borderRadius: 6, border: "none", cursor: "pointer", fontSize: 12,
                background: tab === t ? "#e74c3c" : "#1a1d2e",
                color: tab === t ? "#fff" : "#7f8c8d",
              }}
            >
              {t === "overview" && "Overview"}
              {t === "gene-table" && "Gene Table"}
              {t === "clinical-atlas" && "Clinical Atlas"}
              {t === "definitions" && "Definitions"}
            </button>
          ))}
        </div>

        {/* ── OVERVIEW TAB ── */}
        {tab === "overview" && (
          <div>
            <div style={{ marginBottom: 20 }}>
              <h2 style={{ fontSize: 15, color: "#e74c3c", marginBottom: 12 }}>Gene Risk Comparison</h2>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10 }}>
                {genes.map(g => {
                  const gs = overview?.gene_summary?.[g];
                  return (
                    <div key={g} style={{ background: "#1a1d2e", border: `1px solid ${GENE_COLORS[g]}`, borderRadius: 8, padding: 12 }}>
                      <div style={{ fontSize: 14, fontWeight: 700, color: GENE_COLORS[g], marginBottom: 6 }}>{g}</div>
                      <div style={{ fontSize: 11, color: "#95a5a6", marginBottom: 4 }}>n={gs?.n} · age {gs?.mean_age}yr</div>
                      <div style={{ background: "#0f1117", borderRadius: 4, overflow: "hidden", marginBottom: 4 }}>
                        <div style={{ width: `${gs?.cr_pct}%`, height: 6, background: GENE_COLORS[g] }} />
                      </div>
                      <div style={{ fontSize: 10, color: "#bdc3c7" }}>CR {gs?.cr_pct}% · Relapse {gs?.relapse_pct}%</div>
                      <div style={{ fontSize: 10, color: "#e74c3c", marginTop: 4 }}>RT {gs?.radiation_pct}%</div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div>
              <h2 style={{ fontSize: 15, color: "#e74c3c", marginBottom: 12 }}>Tumour Types by Gene</h2>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                {breakdown?.genes?.map(g => {
                  const bd = breakdown.breakdown?.[g];
                  return (
                    <div key={g} style={{ background: "#1a1d2e", border: `1px solid ${GENE_COLORS[g]}33`, borderRadius: 8, padding: 12 }}>
                      <div style={{ fontSize: 12, fontWeight: 700, color: GENE_COLORS[g], marginBottom: 8 }}>{g} — {GENE_INFO[g]?.locus}</div>
                      {bd?.top_tumour_types?.map((tt, i) => (
                        <div key={i} style={{ fontSize: 11, color: "#bdc3c7", marginBottom: 3 }}>
                          <span style={{ color: GENE_COLORS[g] }}>▸</span> {tt.type} ({tt.count})
                        </div>
                      ))}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* ── GENE TABLE TAB ── */}
        {tab === "gene-table" && (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr style={{ background: "#1a1d2e", color: "#7f8c8d" }}>
                  {["Gene", "Full Name", "Locus", "Size", "Inheritance", "Syndrome", "ACC Risk", "Key Rule"].map(h => (
                    <th key={h} style={{ padding: "8px 10px", textAlign: "left", borderBottom: "1px solid #2c3e50", whiteSpace: "nowrap" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {breakdown?.genes?.map((g, i) => {
                  const bd = breakdown.breakdown?.[g];
                  const gi = GENE_INFO[g];
                  return (
                    <tr key={g} style={{ background: i % 2 === 0 ? "#0f1117" : "#1a1d2e" }}>
                      <td style={{ padding: "8px 10px", color: GENE_COLORS[g], fontWeight: 700 }}>{g}</td>
                      <td style={{ padding: "8px 10px", color: "#ecf0f1" }}>{gi?.full}</td>
                      <td style={{ padding: "8px 10px", color: "#e74c3c" }}>{gi?.locus}</td>
                      <td style={{ padding: "8px 10px", color: "#95a5a6", whiteSpace: "nowrap" }}>{gi?.size}</td>
                      <td style={{ padding: "8px 10px", color: "#f39c12", whiteSpace: "nowrap" }}>{gi?.inh}</td>
                      <td style={{ padding: "8px 10px", color: "#bdc3c7" }}>{bd?.syndrome}</td>
                      <td style={{ padding: "8px 10px", color: "#2ecc71", fontSize: 10 }}>{(bd?.tc_risk || bd?.acc_risk || "")?.substring(0, 55)}…</td>
                      <td style={{ padding: "8px 10px", color: "#e74c3c", fontSize: 10 }}>{bd?.key_rule?.substring(0, 60)}…</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* ── CLINICAL ATLAS TAB ── */}
        {tab === "clinical-atlas" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            {breakdown?.genes?.map(g => {
              const bd = breakdown.breakdown?.[g];
              return (
                <div key={g} style={{ background: "#1a1d2e", border: `1px solid ${GENE_COLORS[g]}`, borderRadius: 10, padding: 16 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
                    <div>
                      <span style={{ fontSize: 15, fontWeight: 700, color: GENE_COLORS[g] }}>{g}</span>
                      <span style={{ fontSize: 11, color: "#95a5a6", marginLeft: 8 }}>{GENE_INFO[g]?.locus} · {GENE_INFO[g]?.size}</span>
                    </div>
                    <span style={{ fontSize: 10, background: "#0f1117", color: "#f39c12", padding: "2px 8px", borderRadius: 4 }}>
                      {GENE_INFO[g]?.inh}
                    </span>
                  </div>
                  <div style={{ fontSize: 11, color: "#bdc3c7", marginBottom: 8 }}>{bd?.syndrome}</div>
                  <div style={{ background: "#0f1117", borderRadius: 6, padding: 10, marginBottom: 8 }}>
                    <div style={{ fontSize: 10, fontWeight: 700, color: "#e67e22", marginBottom: 4 }}>PATHOGNOMONIC</div>
                    <div style={{ fontSize: 11, color: "#ecf0f1" }}>{bd?.pathognomonic}</div>
                  </div>
                  <div style={{ background: "#0f1117", borderRadius: 6, padding: 10, marginBottom: 8 }}>
                    <div style={{ fontSize: 10, fontWeight: 700, color: "#e74c3c", marginBottom: 4 }}>KEY RULE</div>
                    <div style={{ fontSize: 11, color: "#ecf0f1" }}>{bd?.key_rule}</div>
                  </div>
                  <div style={{ marginBottom: 8 }}>
                    <div style={{ fontSize: 10, fontWeight: 700, color: "#e74c3c", marginBottom: 4 }}>SURVEILLANCE</div>
                    {bd?.surveillance_protocols?.map((s, i) => (
                      <div key={i} style={{ fontSize: 10, color: "#95a5a6", marginBottom: 2 }}>• {s}</div>
                    ))}
                  </div>
                  <div>
                    <div style={{ fontSize: 10, fontWeight: 700, color: "#2ecc71", marginBottom: 4 }}>TARGETED Rx</div>
                    <div style={{ fontSize: 11, color: "#bdc3c7" }}>{bd?.targeted_rx}</div>
                  </div>
                  <div style={{ marginTop: 8, display: "flex", gap: 8 }}>
                    <div style={{ fontSize: 10, background: "#0f1117", padding: "2px 8px", borderRadius: 4, color: "#2ecc71" }}>
                      CR {bd?.cr_pct}%
                    </div>
                    <div style={{ fontSize: 10, background: "#0f1117", padding: "2px 8px", borderRadius: 4, color: "#e74c3c" }}>
                      Relapse {bd?.relapse_pct}%
                    </div>
                    <div style={{ fontSize: 10, background: "#0f1117", padding: "2px 8px", borderRadius: 4, color: "#f39c12" }}>
                      RT {bd?.radiation_pct}%
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* ── DEFINITIONS TAB ── */}
        {tab === "definitions" && (
          <div>
            <h2 style={{ fontSize: 15, color: "#e74c3c", marginBottom: 16 }}>Clinical Definitions</h2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 24 }}>
              {definitions?.definitions && Object.entries(definitions.definitions).map(([k, v]) => (
                <div key={k} style={{ background: "#1a1d2e", border: "1px solid #2c3e50", borderRadius: 8, padding: 14 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: "#e74c3c", marginBottom: 8, textTransform: "uppercase" }}>
                    {k.replace(/_/g, " ")}
                  </div>
                  <div style={{ fontSize: 11, color: "#bdc3c7", lineHeight: 1.6 }}>{v}</div>
                </div>
              ))}
            </div>

            <h2 style={{ fontSize: 15, color: "#e74c3c", marginBottom: 16 }}>Key Clinical Rules — Decision-Critical</h2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 10 }}>
              {definitions?.key_clinical_rules?.map((r, i) => (
                <div key={i} style={{ background: "#1a1d2e", border: `1px solid ${GENE_COLORS[r.gene] || "#2c3e50"}`, borderRadius: 8, padding: 14 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                    <div style={{ fontSize: 13, fontWeight: 700, color: GENE_COLORS[r.gene] || "#e74c3c" }}>
                      {r.rule}
                    </div>
                    <span style={{ fontSize: 11, background: "#0f1117", padding: "2px 8px", borderRadius: 4, color: GENE_COLORS[r.gene] || "#95a5a6" }}>
                      {r.gene}
                    </span>
                  </div>
                  <div style={{ fontSize: 11, color: "#95a5a6", marginBottom: 6 }}><strong>Rationale:</strong> {r.rationale}</div>
                  <div style={{ fontSize: 11, color: "#e74c3c" }}><strong>Consequence if missed:</strong> {r.consequence}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Footer */}
        <div style={{ marginTop: 32, paddingTop: 16, borderTop: "1px solid #2c3e50", fontSize: 10, color: "#4a5568", textAlign: "center" }}>
          Hereditary Adrenocortical Carcinoma Predisposition Atlas · 8-Gene Reference (TP53-CTNNB1-CDKN2A-NF1-MEN1-PRKAR1A-ARMC5-MAX) ·
          320 patients (8×40, seeds 3334–3341) · Agenticfinder Expert Dashboards
        </div>
      </div>
    </div>
  );
}
