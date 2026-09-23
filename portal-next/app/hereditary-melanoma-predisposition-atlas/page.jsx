"use client";
import { useEffect, useState } from "react";

const GENE_COLORS = {
  CDKN2A: "#c0392b",
  CDK4:   "#8e44ad",
  BAP1:   "#1abc9c",
  MITF:   "#2980b9",
  POT1:   "#d35400",
  TERT:   "#27ae60",
  MC1R:   "#f39c12",
  NF1:    "#1a6b4a",
};

const GENE_INFO = {
  CDKN2A: { full: "CDKN2A (p16-INK4a+p14-ARF / FAMMM / Melanoma 40-50x RR HIGHEST)", locus: "9p21.3", size: "156 aa / 15 kDa", inh: "AD LOF", risk: "Melanoma 40-50x HIGHEST; pancreatic 20x; multiple primaries PATHOGNOMONIC; annual skin 18yr + EUS 40yr" },
  CDK4:   { full: "CDK4 (Cyclin D1 Kinase / R24C p16-binding abrogated / Melanoma 40-50x)", locus: "12q14.1", size: "303 aa / 34 kDa", inh: "AD GOF", risk: "Melanoma 40-50x RR; CDK4/6 inhibitor paradox; BRAF somatic test mandatory" },
  BAP1:   { full: "BAP1 (Deubiquitinase / BAP1-TPDS / Uveal 50% PATHOGNOMONIC)", locus: "3p21.1", size: "729 aa / 80 kDa", inh: "AD LOF", risk: "Uveal 50% PATHOGNOMONIC; asbestos MANDATORY avoided; BAPomas cascade trigger; tebentafusp FDA2022" },
  MITF:   { full: "MITF (bHLH-LZ Master Melanocyte TF / E318K European founder / Melanoma 5x)", locus: "3p14.1", size: "520 aa / 59 kDa", inh: "AD LOF", risk: "Melanoma 5x; uveal elevated; RCC 3-5x; SUMO acceptor disruption moderate penetrance" },
  POT1:   { full: "POT1 (Telomere OB-fold / Familial Melanoma / Telomere ELONGATION paradox)", locus: "7q31.33", size: "634 aa / 71 kDa", inh: "AD LOF", risk: "Melanoma 4-6x; thyroid 3-5x; glioma 2-3x; LONG telomeres PATHOGNOMONIC (not DC/IPF)" },
  TERT:   { full: "TERT (Telomerase RT / C228T most common somatic melanoma mutation / germline)", locus: "5p15.33", size: "1132 aa / 127 kDa", inh: "AD LOF/promoter", risk: "Somatic C228T/C250T 60-70% melanoma; germline promoter moderate; germline coding LOF = DC (different)" },
  MC1R:   { full: "MC1R (7-TM GPCR / R151C+R160W+D294H / CDKN2A amplifier modifier)", locus: "16q24.3", size: "317 aa / 35 kDa", inh: "AR modifier", risk: "2-3x per high-risk variant; CDKN2A+MC1R compound up to 100x; tanning beds ABSOLUTE CI" },
  NF1:    { full: "NF1 (Neurofibromin RAS-GAP / NF1 / MPNST = SARCOMA not melanoma)", locus: "17q11.2", size: "2839 aa / 319 kDa", inh: "AD LOF", risk: "Melanoma 2-3x; MPNST 8-13% SARCOMA (doxorubicin NOT immunotherapy); selumetinib FDA2020" },
};

export default function HereditaryMelanomaPredispositionAtlasPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [tab, setTab] = useState("overview");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = "/api/hereditary-melanoma-predisposition-atlas";
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-6 text-white">Loading Hereditary Melanoma Predisposition Atlas…</div>;
  if (error)   return <div className="p-6 text-red-400">Error: {error}</div>;

  const genes = overview?.genes || [];

  return (
    <div style={{ background: "#0f1117", minHeight: "100vh", color: "#e0e0e0", fontFamily: "monospace", padding: "24px" }}>
      <div style={{ maxWidth: 1200, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 24, borderBottom: "2px solid #c0392b", paddingBottom: 16 }}>
          <div style={{ fontSize: 11, color: "#7f8c8d", marginBottom: 6 }}>
            🧬 Expert Dashboards → Hereditary Cancer Predisposition Atlases
          </div>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: "#c0392b", margin: 0 }}>
            🔬 Hereditary Melanoma Predisposition Atlas
          </h1>
          <div style={{ fontSize: 12, color: "#95a5a6", marginTop: 6 }}>
            Complete 8-Gene Reference · CDKN2A-CDK4-BAP1-MITF-POT1-TERT-MC1R-NF1 ·
            320-Patient Aggregate (8×40, seeds 3422-3429) · FAMMM / BAP1-TPDS / CDK4 Familial / POT1 / TERT Promoter / MC1R
          </div>
          <div style={{ marginTop: 8, padding: "6px 12px", background: "#1a1a2e", borderRadius: 4, fontSize: 11, color: "#e74c3c", display: "inline-block" }}>
            ⚠ KEY RULES: CDKN2A = melanoma 40-50x HIGHEST + pancreatic 20x | BAP1 = asbestos MANDATORY avoided | POT1 = LONG telomeres (NOT DC/IPF) | TERT C228T = somatic (not germline) | NF1 MPNST = SARCOMA (doxorubicin), NOT immunotherapy
          </div>
        </div>

        {/* Cohort Summary */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 10, marginBottom: 20 }}>
          {[
            { label: "Total Patients", value: overview?.total_patients },
            { label: "Genes", value: (overview?.genes||[]).length },
            { label: "BRAF Somatic %", value: `${overview?.braf_somatic_rate_pct}%` },
            { label: "Immunotherapy %", value: `${overview?.immunotherapy_eligible_rate_pct}%` },
            { label: "Uveal Component %", value: `${overview?.uveal_component_rate_pct}%` },
            { label: "Mean Age Dx", value: overview?.mean_age_at_dx },
          ].map(s => (
            <div key={s.label} style={{ background: "#1a1a2e", borderRadius: 6, padding: "10px 12px", textAlign: "center" }}>
              <div style={{ fontSize: 18, fontWeight: 700, color: "#c0392b" }}>{s.value}</div>
              <div style={{ fontSize: 10, color: "#7f8c8d", marginTop: 2 }}>{s.label}</div>
            </div>
          ))}
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 8, marginBottom: 20, flexWrap: "wrap" }}>
          {["overview", "breakdown", "definitions"].map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              padding: "6px 16px", borderRadius: 4, border: "none", cursor: "pointer", fontSize: 12,
              background: tab === t ? "#c0392b" : "#1a1a2e",
              color: tab === t ? "#fff" : "#95a5a6",
              fontFamily: "monospace",
            }}>{t.charAt(0).toUpperCase() + t.slice(1)}</button>
          ))}
        </div>

        {/* OVERVIEW TAB */}
        {tab === "overview" && (
          <div>
            {/* Gene cards */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 24 }}>
              {genes.map(gene => {
                const info = GENE_INFO[gene] || {};
                const color = GENE_COLORS[gene] || "#888";
                return (
                  <div key={gene} style={{ background: "#1a1a2e", borderRadius: 8, padding: 14, borderLeft: `4px solid ${color}` }}>
                    <div style={{ fontSize: 15, fontWeight: 700, color }}>{gene}</div>
                    <div style={{ fontSize: 10, color: "#95a5a6", marginTop: 2 }}>{info.locus} · {info.size}</div>
                    <div style={{ fontSize: 10, color: "#bdc3c7", marginTop: 4 }}>{info.inh}</div>
                    <div style={{ fontSize: 10, color: "#ecf0f1", marginTop: 6, lineHeight: 1.5 }}>{info.risk}</div>
                  </div>
                );
              })}
            </div>

            {/* Key facts */}
            <div style={{ background: "#1a1a2e", borderRadius: 8, padding: 16, marginBottom: 20 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#c0392b", marginBottom: 10 }}>
                🔑 Key Clinical Facts
              </div>
              {(overview?.key_facts || []).map((fact, i) => (
                <div key={i} style={{ fontSize: 11, color: "#bdc3c7", marginBottom: 6, paddingLeft: 12, borderLeft: "2px solid #c0392b" }}>
                  {fact}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* BREAKDOWN TAB */}
        {tab === "breakdown" && breakdown && (
          <div>
            {(breakdown.genes || genes).map(gene => {
              const bd = breakdown.breakdown?.[gene];
              if (!bd) return null;
              const color = GENE_COLORS[gene] || "#888";
              const info = bd.gene_info || {};
              return (
                <div key={gene} style={{ background: "#1a1a2e", borderRadius: 8, padding: 16, marginBottom: 16, borderLeft: `4px solid ${color}` }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
                    <div>
                      <span style={{ fontSize: 15, fontWeight: 700, color }}>{gene}</span>
                      <span style={{ fontSize: 11, color: "#95a5a6", marginLeft: 10 }}>{info.locus} · {info.inheritance}</span>
                      <div style={{ fontSize: 11, color: "#bdc3c7", marginTop: 3 }}>{info.syndrome}</div>
                    </div>
                    <div style={{ fontSize: 12, color: "#7f8c8d" }}>n={bd.n} · mean age {bd.mean_age}yr</div>
                  </div>

                  {/* Stats grid */}
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 8, marginBottom: 12 }}>
                    {[
                      { label: "BRAF Somatic", value: `${bd.braf_somatic_pct}%` },
                      { label: "Immunotherapy", value: `${bd.immunotherapy_pct}%` },
                      { label: "MPNST Risk", value: `${bd.mpnst_risk_pct}%` },
                      { label: "Uveal Component", value: `${bd.uveal_component_pct}%` },
                      { label: "Multiple Primaries", value: `${bd.multiple_primaries_pct}%` },
                      { label: "Pancreatic Risk", value: `${bd.pancreatic_risk_pct}%` },
                    ].map(s => (
                      <div key={s.label} style={{ background: "#0f1117", borderRadius: 4, padding: "6px 8px", textAlign: "center" }}>
                        <div style={{ fontSize: 13, fontWeight: 700, color }}>{s.value}</div>
                        <div style={{ fontSize: 9, color: "#7f8c8d" }}>{s.label}</div>
                      </div>
                    ))}
                  </div>

                  {/* Top tumour types */}
                  <div style={{ marginBottom: 10 }}>
                    <div style={{ fontSize: 11, color: "#95a5a6", marginBottom: 4 }}>Top tumour types:</div>
                    {(bd.top_tumour_types || []).map(t => (
                      <span key={t.type} style={{ display: "inline-block", background: "#0f1117", borderRadius: 3, padding: "2px 8px", fontSize: 10, color: "#bdc3c7", marginRight: 6, marginBottom: 4 }}>
                        {t.type} ({t.count})
                      </span>
                    ))}
                  </div>

                  {/* Key rules */}
                  <div style={{ fontSize: 11, color: "#e74c3c", background: "#0f1117", borderRadius: 4, padding: "6px 10px", marginBottom: 8 }}>
                    ⚠ {info.key_rule || info.key_avoid}
                  </div>

                  {/* Surveillance */}
                  <div style={{ marginBottom: 8 }}>
                    <div style={{ fontSize: 10, color: "#95a5a6", marginBottom: 3 }}>Surveillance:</div>
                    {(bd.surveillance_protocols || []).map((s, i) => (
                      <div key={i} style={{ fontSize: 10, color: "#bdc3c7", paddingLeft: 8, marginBottom: 1 }}>• {s}</div>
                    ))}
                  </div>

                  {/* Treatment protocols */}
                  <div>
                    <div style={{ fontSize: 10, color: "#95a5a6", marginBottom: 3 }}>Treatment protocols:</div>
                    {(bd.treatment_protocols || []).map((t, i) => (
                      <div key={i} style={{ fontSize: 10, color: "#bdc3c7", paddingLeft: 8, marginBottom: 1 }}>• {t}</div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* DEFINITIONS TAB */}
        {tab === "definitions" && definitions && (
          <div>
            {Object.entries(definitions.definitions || {}).map(([key, text]) => (
              <div key={key} style={{ background: "#1a1a2e", borderRadius: 8, padding: 14, marginBottom: 12, borderLeft: "4px solid #c0392b" }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: "#c0392b", marginBottom: 6 }}>
                  {key.replace(/_/g, " ").toUpperCase()}
                </div>
                <div style={{ fontSize: 11, color: "#bdc3c7", lineHeight: 1.6 }}>{text}</div>
              </div>
            ))}

            {/* Key clinical distinctions */}
            <div style={{ background: "#1a1a2e", borderRadius: 8, padding: 14, marginTop: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#c0392b", marginBottom: 10 }}>
                🔑 Key Clinical Distinctions
              </div>
              {(definitions.key_clinical_distinctions || []).map((d, i) => (
                <div key={i} style={{ fontSize: 11, color: "#bdc3c7", marginBottom: 6, paddingLeft: 12, borderLeft: "2px solid #c0392b" }}>
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
