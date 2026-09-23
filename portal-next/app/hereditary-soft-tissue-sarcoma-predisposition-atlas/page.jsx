"use client";
import { useEffect, useState } from "react";

const GENE_COLORS = {
  TP53:    "#c0392b",
  NF1:     "#8e44ad",
  RB1:     "#1abc9c",
  SMARCB1: "#2980b9",
  DICER1:  "#d35400",
  BRCA2:   "#27ae60",
  EXT1:    "#f39c12",
  SMARCA4: "#16a085",
};

const GENE_INFO = {
  TP53:    { full: "TP53 (p53 / Li-Fraumeni Syndrome / LFS / AVOID RADIATION ABSOLUTELY / WB-MRI Toronto)", locus: "17p13.1", size: "393 aa / 43 kDa", inh: "AD LOF", risk: "Sarcoma 30% lifetime HIGHEST; UPS/RMS/LMS; AVOID RADIATION ABSOLUTELY; WB-MRI Toronto annual MANDATORY" },
  NF1:     { full: "NF1 (Neurofibromin RAS-GAP / MPNST 8-13% HIGHEST / Selumetinib FDA2020 / SARCOMA not glioma)", locus: "17q11.2", size: "2839 aa / 319 kDa", inh: "AD LOF", risk: "MPNST 8-13% HIGHEST hereditary STS; plexiform NF → MPNST; selumetinib FDA2020; doxorubicin/ifosfamide NOT glioma chemo" },
  RB1:     { full: "RB1 (pRb E2F regulator / Secondary STS post-RT 40x / Leiomyosarcoma / CDK4-6i INACTIVE RB1-null)", locus: "13q14.2", size: "928 aa / 105 kDa", inh: "AD LOF", risk: "Secondary STS post-RT 40x HIGHEST; bilateral RB PATHOGNOMONIC; leiomyosarcoma RT field; AVOID high-dose RT germline RB1" },
  SMARCB1: { full: "SMARCB1 (INI1/BAF47 SWI/SNF / RTPS1 / ATRT / Epithelioid Sarcoma INI1-loss / Tazemetostat FDA2020)", locus: "22q11.23", size: "385 aa / 47 kDa", inh: "AD LOF", risk: "RTPS1; ATRT/MRT infants PATHOGNOMONIC; epithelioid sarcoma INI1 IHC loss PATHOGNOMONIC; tazemetostat FDA2020 EZH2i" },
  DICER1:  { full: "DICER1 (RNase III / PPB PATHOGNOMONIC / Embryonal RMS cervix-uterus / CT chest birth-8yr MANDATORY)", locus: "14q32.13", size: "1922 aa / 218 kDa", inh: "AD LOF", risk: "PPB Type I/II/III PATHOGNOMONIC; eRMS cervix/uterus PATHOGNOMONIC; annual low-dose CT chest birth to 8yr MANDATORY siblings" },
  BRCA2:   { full: "BRCA2 (FANCD1 / FA-D1 biallelic / Embryonal RMS PATHOGNOMONIC / AVOID alkylating ABSOLUTELY)", locus: "13q12.3", size: "3418 aa / 384 kDa", inh: "AD LOF / AR biallelic FA", risk: "FA-D1 biallelic: eRMS PATHOGNOMONIC infancy; AVOID alkylating ABSOLUTELY (fatal toxicity FA-D1); sibling exclusion MANDATORY" },
  EXT1:    { full: "EXT1 (Exostosin-1 / HME type 1 / Osteochondromas PATHOGNOMONIC / Chondrosarcoma 1-5% / Cap >2cm alert)", locus: "8q24.11", size: "746 aa / 85 kDa", inh: "AD LOF", risk: "Hereditary multiple exostoses type 1; osteochondromas PATHOGNOMONIC; chondrosarcoma 1-5%; cap >2cm MRI = MALIGNANT alert" },
  SMARCA4: { full: "SMARCA4 (BRG1 SWI/SNF / SCCOHT PATHOGNOMONIC / BRG1 IHC loss / Rhabdoid predisposition RTPS2)", locus: "19p13.2", size: "1647 aa / 185 kDa", inh: "AD LOF", risk: "SCCOHT PATHOGNOMONIC (hypercalcaemia+ovarian); BRG1 IHC loss PATHOGNOMONIC; RTPS2; EZH2i tazemetostat sensitivity" },
};

export default function HereditorySoftTissueSarcomaPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [tab, setTab] = useState("overview");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = "/api/hereditary-soft-tissue-sarcoma-predisposition-atlas";
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-6 text-white">Loading Hereditary Soft Tissue Sarcoma Predisposition Atlas…</div>;
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
            🔬 Hereditary Soft Tissue Sarcoma Predisposition Atlas
          </h1>
          <div style={{ fontSize: 12, color: "#95a5a6", marginTop: 6 }}>
            Complete 8-Gene Reference · TP53-NF1-RB1-SMARCB1-DICER1-BRCA2-EXT1-SMARCA4 ·
            320-Patient Aggregate (8×40, seeds 3438–3445) · LFS / NF1-MPNST / RB1-STS / RTPS1 / DICER1-PPB / FA-D1 / HME / SCCOHT
          </div>
          <div style={{ marginTop: 8, padding: "6px 12px", background: "#1a1a2e", borderRadius: 4, fontSize: 11, color: "#e74c3c", display: "inline-block" }}>
            ⚠ KEY RULES: TP53/LFS = AVOID RADIATION ABSOLUTELY + WB-MRI Toronto | NF1 MPNST = SARCOMA (doxorubicin NOT glioma) | SMARCB1 INI1-loss = tazemetostat FDA2020 | DICER1 = CT chest birth-8yr MANDATORY | BRCA2 FA-D1 = AVOID alkylating ABSOLUTELY | EXT1 cap &gt;2cm = malignant alert
          </div>
        </div>

        {/* Cohort Summary */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 10, marginBottom: 20 }}>
          {[
            { label: "Total Patients", value: overview?.total_patients },
            { label: "Genes", value: (overview?.genes||[]).length },
            { label: "Radiation CI %", value: overview?.radiation_contraindicated_rate_pct != null ? `${overview.radiation_contraindicated_rate_pct}%` : "—" },
            { label: "Targeted Rx %", value: overview?.targeted_therapy_rate_pct != null ? `${overview.targeted_therapy_rate_pct}%` : "—" },
            { label: "INI1-loss %", value: overview?.ini1_loss_rate_pct != null ? `${overview.ini1_loss_rate_pct}%` : "—" },
            { label: "Mean Age Dx", value: overview?.mean_age_at_dx != null ? `${overview.mean_age_at_dx}yr` : "—" },
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

                  <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 8, marginBottom: 12 }}>
                    {[
                      { label: "Radiation CI", value: `${bd.radiation_contraindicated_pct ?? "—"}%` },
                      { label: "Targeted Rx", value: `${bd.targeted_therapy_pct ?? "—"}%` },
                      { label: "MPNST Risk", value: `${bd.mpnst_risk_pct ?? "—"}%` },
                      { label: "INI1 Loss", value: `${bd.ini1_loss_pct ?? "—"}%` },
                      { label: "Alkylator CI", value: `${bd.alkylator_ci_pct ?? "—"}%` },
                    ].map(s => (
                      <div key={s.label} style={{ background: "#0f1117", borderRadius: 4, padding: "6px 8px", textAlign: "center" }}>
                        <div style={{ fontSize: 13, fontWeight: 700, color }}>{s.value}</div>
                        <div style={{ fontSize: 9, color: "#7f8c8d" }}>{s.label}</div>
                      </div>
                    ))}
                  </div>

                  <div style={{ marginBottom: 10 }}>
                    <div style={{ fontSize: 11, color: "#95a5a6", marginBottom: 4 }}>Top tumour types:</div>
                    {(bd.top_tumour_types || []).map(t => (
                      <span key={t.type} style={{ display: "inline-block", background: "#0f1117", borderRadius: 3, padding: "2px 8px", fontSize: 10, color: "#bdc3c7", marginRight: 6, marginBottom: 4 }}>
                        {t.type} ({t.count})
                      </span>
                    ))}
                  </div>

                  <div style={{ fontSize: 11, color: "#e74c3c", background: "#0f1117", borderRadius: 4, padding: "6px 10px", marginBottom: 8 }}>
                    ⚠ {info.key_rule || info.key_avoid}
                  </div>

                  <div style={{ marginBottom: 8 }}>
                    <div style={{ fontSize: 10, color: "#95a5a6", marginBottom: 3 }}>Surveillance:</div>
                    {(bd.surveillance_protocols || []).map((s, i) => (
                      <div key={i} style={{ fontSize: 10, color: "#bdc3c7", paddingLeft: 8, marginBottom: 1 }}>• {s}</div>
                    ))}
                  </div>

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
