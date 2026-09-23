"use client";
import { useEffect, useState } from "react";

const GENE_COLORS = {
  MEN1:    "#1a6b4a",
  VHL:     "#2980b9",
  RET:     "#c0392b",
  NF1:     "#8e44ad",
  TSC2:    "#d35400",
  CDKN1B:  "#16a085",
  PRKAR1A: "#f39c12",
  SDHB:    "#27ae60",
};

const GENE_INFO = {
  MEN1:    { full: "MEN1 (Menin Nuclear Scaffold / MEN1 Syndrome / pNETs+HPT+Pituitary)", locus: "11q13.1", size: "610 aa / 68 kDa", inh: "AD LOF", risk: "pNETs 40-70%; gastrinoma ZES PATHOGNOMONIC" },
  VHL:     { full: "VHL (HIF-α Substrate Adaptor / VHL Disease / Clear-Cell pNETs Only)", locus: "3p25.3", size: "213 aa / 24 kDa", inh: "AD LOF", risk: "pNETs 15-17% clear cell non-functional ONLY" },
  RET:     { full: "RET (Receptor Tyrosine Kinase GOF / MEN2A-MEN2B / MTC 100%)", locus: "10q11.21", size: "1114 aa / 124 kDa", inh: "AD GOF", risk: "MTC 100% penetrance; pheo 50% bilateral" },
  NF1:     { full: "NF1 (Neurofibromin RAS-GAP / NF1 / Duodenal Somatostatinoma PATHOGNOMONIC)", locus: "17q11.2", size: "2839 aa / 319 kDa", inh: "AD LOF", risk: "Duodenal NETs; MPNST 8-13%; GIST 7%" },
  TSC2:    { full: "TSC2 (Tuberin mTOR-GAP / TSC / Everolimus Direct mTOR Target)", locus: "16p13.3", size: "1807 aa / 198 kDa", inh: "AD LOF", risk: "pNETs rare; pulmonary carcinoids; AML 80%" },
  CDKN1B:  { full: "CDKN1B (p27/KIP1 CDK2 Inhibitor / MEN4 / Exclude MEN1 First)", locus: "12p13.1", size: "196 aa / 22 kDa", inh: "AD LOF", risk: "pituitary + parathyroid + pNETs; MEN4 rare" },
  PRKAR1A: { full: "PRKAR1A (PKA Regulatory Subunit / Carney Complex / Cardiac Myxoma LIFE-THREATENING)", locus: "17q24.2", size: "381 aa / 43 kDa", inh: "AD LOF", risk: "PPNAD; cardiac myxoma MANDATORY echo; acromegaly" },
  SDHB:    { full: "SDHB (SDH B Iron-Sulfur / PGL/PHEO / Malignant PGL 35-40% HIGHEST)", locus: "1p36.13", size: "280 aa / 32 kDa", inh: "AD LOF (AR biallelic full loss)", risk: "Malignant PGL 35-40% HIGHEST; pNETs 10-15%" },
};

export default function HereditaryNETCarcinoidPredispositionAtlasPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [tab, setTab] = useState("overview");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = "/api/hereditary-net-carcinoid-predisposition-atlas";
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-6 text-white">Loading Hereditary NET/Carcinoid Predisposition Atlas…</div>;
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
            🏥 Hereditary NET / Carcinoid Predisposition Atlas
          </h1>
          <div style={{ fontSize: 12, color: "#95a5a6", marginTop: 6 }}>
            Complete 8-Gene Reference · MEN1-VHL-RET-NF1-TSC2-CDKN1B-PRKAR1A-SDHB ·
            320-Patient Aggregate (8×40, seeds 3406-3413) · MEN1 / MEN2 / VHL / NF1 / TSC / Carney / SDHx
          </div>
          <div style={{ marginTop: 8, padding: "6px 12px", background: "#1a1a2e", borderRadius: 4, fontSize: 11, color: "#e74c3c", display: "inline-block" }}>
            ⚠ KEY RULES: MEN1 = concurrent HPT+pituitary+pNET | VHL pNETs = clear cell non-functional ONLY | RET = codon-based thyroidectomy timing | PRKAR1A = cardiac myxoma annual echo MANDATORY | SDHB = malignant PGL 35-40%
          </div>
        </div>

        {/* Cohort Summary */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 10, marginBottom: 20 }}>
          {[
            { label: "Total Patients", value: overview?.total_patients },
            { label: "Genes", value: (overview?.genes||[]).length },
            { label: "Somatostatin Eligible %", value: `${overview?.somatostatin_eligible_rate_pct}%` },
            { label: "mTOR Eligible %", value: `${overview?.mtor_eligible_rate_pct}%` },
            { label: "Malignant PGL %", value: `${overview?.malignant_pgl_rate_pct}%` },
            { label: "Mean Age Dx", value: overview?.mean_age_at_dx },
          ].map(s => (
            <div key={s.label} style={{ background: "#1a1a2e", borderRadius: 6, padding: "10px 12px", textAlign: "center" }}>
              <div style={{ fontSize: 18, fontWeight: 700, color: "#1a6b4a" }}>{s.value}</div>
              <div style={{ fontSize: 10, color: "#95a5a6", marginTop: 2 }}>{s.label}</div>
            </div>
          ))}
        </div>

        {/* Gene Cards */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 20 }}>
          {genes.map(gene => {
            const info = GENE_INFO[gene] || {};
            const cnt = overview?.gene_counts?.[gene] || 0;
            return (
              <div key={gene} style={{ background: "#1a1a2e", borderRadius: 6, padding: "10px 12px", borderLeft: `3px solid ${GENE_COLORS[gene] || "#555"}` }}>
                <div style={{ fontSize: 14, fontWeight: 700, color: GENE_COLORS[gene] || "#aaa" }}>{gene}</div>
                <div style={{ fontSize: 10, color: "#95a5a6", marginTop: 2 }}>{info.locus} · {info.size}</div>
                <div style={{ fontSize: 10, color: "#7f8c8d", marginTop: 2 }}>{info.inh}</div>
                <div style={{ fontSize: 10, color: "#bdc3c7", marginTop: 4 }}>{info.risk}</div>
                <div style={{ fontSize: 10, color: "#95a5a6", marginTop: 4 }}>n = {cnt} patients</div>
              </div>
            );
          })}
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
          {["overview", "breakdown", "definitions"].map(t => (
            <button key={t} onClick={() => setTab(t)}
              style={{ padding: "6px 14px", borderRadius: 4, border: "none", cursor: "pointer", fontSize: 12, fontWeight: 600,
                background: tab === t ? "#1a6b4a" : "#1a1a2e", color: tab === t ? "#fff" : "#95a5a6" }}>
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        {tab === "overview" && overview && (
          <div>
            <div style={{ background: "#1a1a2e", borderRadius: 6, padding: 16, marginBottom: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#1a6b4a", marginBottom: 10 }}>Seed Range: {overview.seed_range}</div>
              <div style={{ fontSize: 12, color: "#95a5a6", marginBottom: 8 }}>Key Clinical Facts:</div>
              {(overview.key_facts || []).map((f, i) => (
                <div key={i} style={{ fontSize: 11, color: "#bdc3c7", marginBottom: 6, paddingLeft: 10, borderLeft: "2px solid #1a6b4a" }}>
                  {f}
                </div>
              ))}
            </div>
            <div style={{ background: "#1a1a2e", borderRadius: 6, padding: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#1a6b4a", marginBottom: 10 }}>Gene Distribution (n=40 per gene)</div>
              {genes.map(gene => (
                <div key={gene} style={{ display: "flex", alignItems: "center", marginBottom: 8 }}>
                  <div style={{ width: 80, fontSize: 11, color: GENE_COLORS[gene] || "#aaa", fontWeight: 700 }}>{gene}</div>
                  <div style={{ flex: 1, height: 16, background: "#0f1117", borderRadius: 3, overflow: "hidden" }}>
                    <div style={{ height: "100%", width: `${((overview.gene_counts?.[gene] || 0) / overview.total_patients) * 100}%`, background: GENE_COLORS[gene] || "#555", borderRadius: 3 }} />
                  </div>
                  <div style={{ width: 40, fontSize: 11, color: "#95a5a6", textAlign: "right" }}>{overview.gene_counts?.[gene] || 0}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {tab === "breakdown" && breakdown && (
          <div>
            {genes.map(gene => {
              const gd = breakdown.breakdown?.[gene];
              if (!gd) return null;
              return (
                <div key={gene} style={{ background: "#1a1a2e", borderRadius: 6, padding: 16, marginBottom: 12, borderLeft: `3px solid ${GENE_COLORS[gene] || "#555"}` }}>
                  <div style={{ fontSize: 14, fontWeight: 700, color: GENE_COLORS[gene] || "#aaa", marginBottom: 8 }}>{gene}</div>
                  <div style={{ fontSize: 11, color: "#95a5a6", marginBottom: 6 }}>{gd.gene_info?.syndrome}</div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 8, marginBottom: 12 }}>
                    {[
                      { label: "n", value: gd.n },
                      { label: "SSA Eligible %", value: `${gd.somatostatin_eligible_pct}%` },
                      { label: "mTOR Eligible %", value: `${gd.mtor_eligible_pct}%` },
                      { label: "Mean Age Dx", value: gd.mean_age },
                    ].map(m => (
                      <div key={m.label} style={{ background: "#0f1117", borderRadius: 4, padding: "6px 10px", textAlign: "center" }}>
                        <div style={{ fontSize: 14, fontWeight: 700, color: GENE_COLORS[gene] || "#aaa" }}>{m.value}</div>
                        <div style={{ fontSize: 10, color: "#7f8c8d" }}>{m.label}</div>
                      </div>
                    ))}
                  </div>
                  <div style={{ marginBottom: 8 }}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: "#bdc3c7", marginBottom: 4 }}>Top Tumour Types:</div>
                    {(gd.top_tumour_types || []).map((t, i) => (
                      <div key={i} style={{ fontSize: 10, color: "#95a5a6", marginBottom: 2 }}>• {t.type} (n={t.count})</div>
                    ))}
                  </div>
                  <div style={{ marginBottom: 8 }}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: "#bdc3c7", marginBottom: 4 }}>Treatment Protocols:</div>
                    {(gd.treatment_protocols || []).map((p, i) => (
                      <div key={i} style={{ fontSize: 10, color: "#95a5a6", marginBottom: 2 }}>• {p}</div>
                    ))}
                  </div>
                  <div>
                    <div style={{ fontSize: 11, fontWeight: 700, color: "#bdc3c7", marginBottom: 4 }}>Surveillance Protocols:</div>
                    {(gd.surveillance_protocols || []).map((p, i) => (
                      <div key={i} style={{ fontSize: 10, color: "#95a5a6", marginBottom: 2 }}>• {p}</div>
                    ))}
                  </div>
                  <div style={{ marginTop: 8, fontSize: 10, color: "#7f8c8d", borderTop: "1px solid #2c3e50", paddingTop: 6 }}>
                    <strong>Key Rule:</strong> {gd.gene_info?.key_rule}
                  </div>
                  <div style={{ fontSize: 10, color: "#e74c3c", marginTop: 4 }}>
                    <strong>⚠ Avoid:</strong> {gd.gene_info?.key_avoid}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {tab === "definitions" && definitions && (
          <div>
            <div style={{ background: "#1a1a2e", borderRadius: 6, padding: 16, marginBottom: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#1a6b4a", marginBottom: 12 }}>Key Clinical Distinctions</div>
              {(definitions.key_clinical_distinctions || []).map((d, i) => (
                <div key={i} style={{ fontSize: 11, color: "#bdc3c7", marginBottom: 8, paddingLeft: 10, borderLeft: "2px solid #1a6b4a" }}>
                  {d}
                </div>
              ))}
            </div>
            {Object.entries(definitions.definitions || {}).map(([key, val]) => (
              <div key={key} style={{ background: "#1a1a2e", borderRadius: 6, padding: 14, marginBottom: 10 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: "#1a6b4a", marginBottom: 6, textTransform: "uppercase" }}>{key.replace(/_/g, " ")}</div>
                <div style={{ fontSize: 11, color: "#bdc3c7", lineHeight: 1.6 }}>{val}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
