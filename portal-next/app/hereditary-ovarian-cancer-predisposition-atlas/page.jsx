"use client";
import { useEffect, useState } from "react";

const GENE_COLORS = {
  BRCA1:  "#1a6b4a",
  BRCA2:  "#c0392b",
  BRIP1:  "#2980b9",
  RAD51C: "#8e44ad",
  RAD51D: "#d35400",
  PALB2:  "#16a085",
  MLH1:   "#f39c12",
  STK11:  "#27ae60",
};

const GENE_INFO = {
  BRCA1:  { full: "Breast Cancer Gene 1 (RING-BRCT HR Scaffold)", locus: "17q21.31", size: "1863 aa / 208 kDa", inh: "AD LOF", risk: "39-44%" },
  BRCA2:  { full: "Breast Cancer Gene 2 (RAD51 Mediator / FANCD1)", locus: "13q12.3",  size: "3418 aa / 384 kDa", inh: "AD LOF", risk: "11-17%" },
  BRIP1:  { full: "BRCA1-Interacting Helicase (FANCJ / 5'-3' Helicase)", locus: "17q23.2", size: "1249 aa / 140 kDa", inh: "AD LOF", risk: "5-8×" },
  RAD51C: { full: "RAD51 Paralog C (FANCO / BCDX2+CX3 Complexes)", locus: "17q22",    size: "376 aa / 40 kDa",   inh: "AD LOF", risk: "~6%" },
  RAD51D: { full: "RAD51 Paralog D (BCDX2 Complex / PARP-Sensitive)", locus: "17q12",   size: "328 aa / 37 kDa",   inh: "AD LOF", risk: "~10%" },
  PALB2:  { full: "Partner and Localiser of BRCA2 (FANCN / BRCA1-BRCA2 Bridge)", locus: "16p12.2", size: "1186 aa / 131 kDa", inh: "AD LOF", risk: "3-5% OCA / 53% breast" },
  MLH1:   { full: "MutL Homolog 1 (MutLα Scaffold / Lynch Type 1)", locus: "3p22.2",   size: "756 aa / 85 kDa",   inh: "AD LOF", risk: "8-13% endometrioid" },
  STK11:  { full: "Serine/Threonine Kinase 11 (LKB1 / AMPK Master Kinase)", locus: "19p13.3", size: "433 aa / 48 kDa", inh: "AD LOF", risk: "21% SCTAT" },
};

export default function HereditaryOvarianCancerPredispositionAtlasPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [tab, setTab] = useState("overview");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = "/api/hereditary-ovarian-cancer-predisposition-atlas";
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-6 text-white">Loading Hereditary Ovarian Cancer Predisposition Atlas…</div>;
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
            🏥 Hereditary Ovarian Cancer Predisposition Atlas
          </h1>
          <div style={{ fontSize: 12, color: "#95a5a6", marginTop: 6 }}>
            Complete 8-Gene Reference · BRCA1-BRCA2-BRIP1-RAD51C-RAD51D-PALB2-MLH1-STK11
            · 320-Patient Aggregate (seeds 3374–3381)
          </div>
        </div>

        {/* KPI row */}
        {overview && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 12, marginBottom: 24 }}>
            {[
              { label: "Total Patients", value: overview.total_patients },
              { label: "PARPi Rate", value: `${overview.parp_inhibitor_rate_pct}%` },
              { label: "Platinum Response", value: `${overview.platinum_response_rate_pct}%` },
              { label: "RRSO Performed", value: `${overview.rrso_rate_pct}%` },
              { label: "Stage III/IV", value: `${overview.stage_iii_iv_pct}%` },
            ].map(k => (
              <div key={k.label} style={{ background: "#1a1d2e", borderRadius: 8, padding: "12px 16px", textAlign: "center" }}>
                <div style={{ fontSize: 22, fontWeight: 700, color: "#1a6b4a" }}>{k.value}</div>
                <div style={{ fontSize: 11, color: "#7f8c8d", marginTop: 4 }}>{k.label}</div>
              </div>
            ))}
          </div>
        )}

        {/* Tabs */}
        <div style={{ display: "flex", gap: 8, marginBottom: 20, flexWrap: "wrap" }}>
          {["overview", "breakdown", "definitions"].map(t => (
            <button key={t} onClick={() => setTab(t)}
              style={{ padding: "6px 18px", borderRadius: 6, border: "none", cursor: "pointer",
                background: tab === t ? "#1a6b4a" : "#1a1d2e", color: tab === t ? "#fff" : "#95a5a6",
                fontFamily: "monospace", fontSize: 13, fontWeight: tab === t ? 700 : 400 }}>
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>

        {/* OVERVIEW TAB */}
        {tab === "overview" && overview && (
          <div>
            {/* Gene grid */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 24 }}>
              {genes.map(gene => {
                const info = GENE_INFO[gene] || {};
                const count = overview.gene_counts?.[gene] || 0;
                return (
                  <div key={gene} style={{ background: "#1a1d2e", borderRadius: 8, padding: 14,
                    borderLeft: `4px solid ${GENE_COLORS[gene] || "#555"}` }}>
                    <div style={{ fontSize: 16, fontWeight: 700, color: GENE_COLORS[gene] || "#fff" }}>{gene}</div>
                    <div style={{ fontSize: 10, color: "#7f8c8d", marginTop: 2 }}>{info.locus} · {info.size}</div>
                    <div style={{ fontSize: 10, color: "#bdc3c7", marginTop: 4 }}>{info.full}</div>
                    <div style={{ fontSize: 10, color: "#e74c3c", marginTop: 4, fontWeight: 700 }}>
                      OCA Risk: {info.risk}
                    </div>
                    <div style={{ fontSize: 10, color: "#95a5a6", marginTop: 2 }}>
                      {info.inh} · n={count}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Key facts */}
            <div style={{ background: "#1a1d2e", borderRadius: 8, padding: 16, marginBottom: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#1a6b4a", marginBottom: 10 }}>
                ⚠ Key Clinical Facts — Hereditary Ovarian Cancer Predisposition
              </div>
              {(overview.key_facts || []).map((f, i) => (
                <div key={i} style={{ fontSize: 12, color: "#bdc3c7", marginBottom: 6, paddingLeft: 12,
                  borderLeft: "2px solid #1a6b4a" }}>
                  {f}
                </div>
              ))}
            </div>

            {/* RRSO timing table */}
            <div style={{ background: "#1a1d2e", borderRadius: 8, padding: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#1a6b4a", marginBottom: 10 }}>
                🔑 RRSO Timing Guide (Critical Distinction)
              </div>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid #2c3e50" }}>
                    {["Gene", "Ovarian Risk", "RRSO Age", "Breast Risk", "Key Note"].map(h => (
                      <th key={h} style={{ textAlign: "left", padding: "6px 10px", color: "#7f8c8d" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["BRCA1",  "39-44%",  "35-40yr",  "72%",     "HGSOC fallopian tube origin; RRSO 35-40yr MANDATORY"],
                    ["BRCA2",  "11-17%",  "40-45yr",  "65%",     "Later onset ~55yr; male breast 6%; prostate 24-40%"],
                    ["BRIP1",  "5-8×",    "45-50yr",  "None",    "NO breast risk — key distinction; FANCJ biallelic"],
                    ["RAD51C", "~6%",     "45-50yr",  "None",    "NO breast risk; FANCO biallelic; HRD PARPi emerging"],
                    ["RAD51D", "~10%",    "45-50yr",  "None",    "HIGHEST paralog; PARPi sensitivity established"],
                    ["PALB2",  "3-5%",    "45-50yr*", "53%",     "Breast DOMINATES; *breast-driven RRSO often earlier"],
                    ["MLH1",   "8-13%",   "35-40yr",  "Low",     "ENDOMETRIOID not HGSOC; pembrolizumab dMMR/MSI-H"],
                    ["STK11",  "21%",     "Post-cb",  "50%",     "SCTAT PATHOGNOMONIC; adenoma malignum cervix"],
                  ].map(([gene, risk, rrso, breast, note]) => (
                    <tr key={gene} style={{ borderBottom: "1px solid #1a1d2e" }}>
                      <td style={{ padding: "6px 10px", color: GENE_COLORS[gene] || "#fff", fontWeight: 700 }}>{gene}</td>
                      <td style={{ padding: "6px 10px", color: "#e74c3c" }}>{risk}</td>
                      <td style={{ padding: "6px 10px", color: "#f39c12" }}>{rrso}</td>
                      <td style={{ padding: "6px 10px", color: "#2980b9" }}>{breast}</td>
                      <td style={{ padding: "6px 10px", color: "#bdc3c7", fontSize: 10 }}>{note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* BREAKDOWN TAB */}
        {tab === "breakdown" && breakdown && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {(breakdown.genes || []).map(gene => {
              const b = breakdown.breakdown?.[gene];
              if (!b) return null;
              const gi = b.gene_info;
              return (
                <div key={gene} style={{ background: "#1a1d2e", borderRadius: 8, padding: 16,
                  borderLeft: `4px solid ${GENE_COLORS[gene] || "#555"}` }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
                    <div>
                      <span style={{ fontSize: 16, fontWeight: 700, color: GENE_COLORS[gene] || "#fff" }}>{gene}</span>
                      <span style={{ fontSize: 11, color: "#7f8c8d", marginLeft: 10 }}>{gi?.locus} · {gi?.inheritance}</span>
                      <div style={{ fontSize: 11, color: "#bdc3c7", marginTop: 2 }}>{gi?.syndrome}</div>
                      <div style={{ fontSize: 11, color: "#e74c3c", marginTop: 2 }}>Ovarian risk: {gi?.ovarian_risk}</div>
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 8, textAlign: "center" }}>
                      {[
                        ["PARPi", `${b.parp_inhibitor_pct}%`, "#1a6b4a"],
                        ["Pt Resp", `${b.platinum_response_pct}%`, "#2980b9"],
                        ["RRSO", `${b.rrso_pct}%`, "#f39c12"],
                        ["Relapse", `${b.relapse_pct}%`, "#e74c3c"],
                      ].map(([l, v, c]) => (
                        <div key={l} style={{ background: "#0f1117", borderRadius: 6, padding: "8px 12px" }}>
                          <div style={{ fontSize: 16, fontWeight: 700, color: c }}>{v}</div>
                          <div style={{ fontSize: 10, color: "#7f8c8d" }}>{l}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, fontSize: 11 }}>
                    <div>
                      <div style={{ color: "#f39c12", fontWeight: 700, marginBottom: 4 }}>Key Rule:</div>
                      <div style={{ color: "#bdc3c7" }}>{gi?.key_rule}</div>
                      <div style={{ color: "#e74c3c", fontWeight: 700, marginTop: 8, marginBottom: 4 }}>⚠ Key Avoid:</div>
                      <div style={{ color: "#bdc3c7" }}>{gi?.key_avoid}</div>
                    </div>
                    <div>
                      <div style={{ color: "#2980b9", fontWeight: 700, marginBottom: 4 }}>Targeted Rx:</div>
                      <div style={{ color: "#bdc3c7" }}>{gi?.targeted_rx}</div>
                      <div style={{ color: "#1a6b4a", fontWeight: 700, marginTop: 8, marginBottom: 4 }}>Surveillance:</div>
                      <div style={{ color: "#bdc3c7", fontSize: 10 }}>
                        {(b.surveillance_protocols || []).join(" · ")}
                      </div>
                    </div>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 10 }}>
                    <div>
                      <div style={{ fontSize: 10, color: "#7f8c8d", marginBottom: 4 }}>Top Tumour Types</div>
                      {(b.top_tumour_types || []).map(t => (
                        <div key={t.type} style={{ fontSize: 11, color: "#bdc3c7" }}>
                          {t.type}: <span style={{ color: GENE_COLORS[gene] }}>{t.count}</span>
                        </div>
                      ))}
                    </div>
                    <div>
                      <div style={{ fontSize: 10, color: "#7f8c8d", marginBottom: 4 }}>Top Variants</div>
                      {(b.top_variants || []).map(v => (
                        <div key={v.variant} style={{ fontSize: 10, color: "#95a5a6" }}>
                          {v.variant} ({v.count})
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* DEFINITIONS TAB */}
        {tab === "definitions" && definitions && (
          <div>
            <div style={{ display: "flex", flexDirection: "column", gap: 12, marginBottom: 20 }}>
              {Object.entries(definitions.definitions || {}).map(([key, val]) => (
                <div key={key} style={{ background: "#1a1d2e", borderRadius: 8, padding: 14,
                  borderLeft: "4px solid #1a6b4a" }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: "#1a6b4a", marginBottom: 6 }}>
                    {key.replace(/_/g, " ").toUpperCase()}
                  </div>
                  <div style={{ fontSize: 12, color: "#bdc3c7", lineHeight: 1.6 }}>{val}</div>
                </div>
              ))}
            </div>
            <div style={{ background: "#1a1d2e", borderRadius: 8, padding: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#1a6b4a", marginBottom: 10 }}>
                Key Clinical Distinctions
              </div>
              {(definitions.key_clinical_distinctions || []).map((d, i) => (
                <div key={i} style={{ fontSize: 12, color: "#bdc3c7", marginBottom: 8, paddingLeft: 12,
                  borderLeft: "2px solid #2980b9" }}>
                  {d}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
