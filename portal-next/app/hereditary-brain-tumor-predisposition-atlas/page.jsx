"use client";
import { useEffect, useState } from "react";

const GENE_COLORS = {
  TP53:  "#c0392b",
  NF1:   "#8e44ad",
  NF2:   "#1abc9c",
  VHL:   "#2980b9",
  PTCH1: "#d35400",
  TSC2:  "#27ae60",
  PTEN:  "#f39c12",
  SUFU:  "#1a6b4a",
};

const GENE_INFO = {
  TP53:  { full: "TP53 (p53 / Li-Fraumeni Syndrome / LFS / AVOID RADIATION ABSOLUTELY / WB-MRI Toronto)", locus: "17p13.1", size: "393 aa / 43 kDa", inh: "AD LOF", risk: "Brain tumors 20-26% (GBM/DIPG/astrocytoma); AVOID RADIATION ABSOLUTELY; WB-MRI Toronto annual MANDATORY; ONC201 H3K27M FDA2022" },
  NF1:   { full: "NF1 (Neurofibromin RAS-GAP / Optic Pathway Glioma 15-20% PATHOGNOMONIC / Selumetinib FDA2020)", locus: "17q11.2", size: "2839 aa / 319 kDa", inh: "AD LOF", risk: "OPG 15-20% PATHOGNOMONIC; café-au-lait 6+ PATHOGNOMONIC; MPNST 8-13% SARCOMA (doxorubicin NOT glioma); selumetinib FDA2020" },
  NF2:   { full: "NF2 (Merlin/Schwannomin / Bilateral VS PATHOGNOMONIC / Bevacizumab / Meningioma)", locus: "22q12.2", size: "595 aa / 66 kDa", inh: "AD LOF", risk: "Bilateral VS 90-95% PATHOGNOMONIC; meningioma 50-80%; ependymoma 30-53%; cataract 80% PATHOGNOMONIC; bevacizumab hearing" },
  VHL:   { full: "VHL (HIF-2alpha / CNS Hemangioblastoma PATHOGNOMONIC / Retinal Angioma / Belzutifan FDA2021)", locus: "3p25.3", size: "213 aa / 24 kDa", inh: "AD LOF", risk: "CNS hemangioblastoma 60-80% PATHOGNOMONIC; retinal angioma 50-60% PATHOGNOMONIC; ELST hearing; belzutifan FDA2021" },
  PTCH1: { full: "PTCH1 (Gorlin/NBCCS / Desmoplastic MB 5% / BCC 1000s / OKC PATHOGNOMONIC / AVOID RADIATION ABSOLUTELY)", locus: "9q22.32", size: "1447 aa / 161 kDa", inh: "AD LOF", risk: "Desmoplastic MB 5% first 5yr; AVOID RADIATION ABSOLUTELY (1000s BCCs field); OKC PATHOGNOMONIC; vismodegib BCC FDA2012" },
  TSC2:  { full: "TSC2 (Tuberin mTOR-GAP / SEGA PATHOGNOMONIC / Cortical Tubers PATHOGNOMONIC / Everolimus FDA2012)", locus: "16p13.3", size: "1807 aa / 198 kDa", inh: "AD LOF", risk: "SEGA foramen Monro PATHOGNOMONIC; cortical tubers PATHOGNOMONIC; everolimus FDA2012 SEGA; vigabatrin infantile spasms" },
  PTEN:  { full: "PTEN (Cowden/PHTS / Lhermitte-Duclos PATHOGNOMONIC / Macrocephaly 90% / Breast 85% HIGHEST)", locus: "10q23.31", size: "403 aa / 47 kDa", inh: "AD LOF", risk: "LDD cerebellar gangliocytoma PATHOGNOMONIC (striped MRI); macrocephaly 90% PATHOGNOMONIC; breast 85% annual MRI 25-30yr" },
  SUFU:  { full: "SUFU (SHH Pathway Suppressor / Medulloblastoma 33% HIGHEST / Gorlin-Like / Meningioma)", locus: "10q24.32", size: "484 aa / 54 kDa", inh: "AD LOF", risk: "SHH MB 33% HIGHEST germline SHH gene; desmoplastic nodular MB PATHOGNOMONIC; MRI from BIRTH; AVOID radiation infants" },
};

export default function HereditaryBrainTumorPredispositionAtlasPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [tab, setTab] = useState("overview");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = "/api/hereditary-brain-tumor-predisposition-atlas";
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-6 text-white">Loading Hereditary Brain Tumor Predisposition Atlas…</div>;
  if (error)   return <div className="p-6 text-red-400">Error: {error}</div>;

  const genes = overview?.genes || [];

  return (
    <div style={{ background: "#0f1117", minHeight: "100vh", color: "#e0e0e0", fontFamily: "monospace", padding: "24px" }}>
      <div style={{ maxWidth: 1200, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 24, borderBottom: "2px solid #2980b9", paddingBottom: 16 }}>
          <div style={{ fontSize: 11, color: "#7f8c8d", marginBottom: 6 }}>
            🧬 Expert Dashboards → Hereditary Cancer Predisposition Atlases
          </div>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: "#2980b9", margin: 0 }}>
            🔬 Hereditary Brain Tumor Predisposition Atlas
          </h1>
          <div style={{ fontSize: 12, color: "#95a5a6", marginTop: 6 }}>
            Complete 8-Gene Reference · TP53-NF1-NF2-VHL-PTCH1-TSC2-PTEN-SUFU ·
            320-Patient Aggregate (8×40, seeds 3430-3437) · LFS / NF1 / NF2 / VHL / Gorlin / TSC / Cowden / SUFU-SHH
          </div>
          <div style={{ marginTop: 8, padding: "6px 12px", background: "#1a1a2e", borderRadius: 4, fontSize: 11, color: "#e74c3c", display: "inline-block" }}>
            ⚠ KEY RULES: TP53/LFS = AVOID RADIATION ABSOLUTELY + WB-MRI Toronto | NF1 MPNST = SARCOMA (doxorubicin NOT glioma) | PTCH1/Gorlin = AVOID RADIATION (1000s BCCs) | SUFU = MRI from BIRTH | TSC2 SEGA = Everolimus FDA2012 | VHL hemangioblastoma = Belzutifan FDA2021
          </div>
        </div>

        {/* Cohort Summary */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 10, marginBottom: 20 }}>
          {[
            { label: "Total Patients", value: overview?.total_patients },
            { label: "Genes", value: (overview?.genes||[]).length },
            { label: "Radiation CI %", value: `${overview?.radiation_contraindicated_rate_pct}%` },
            { label: "Targeted Therapy %", value: `${overview?.targeted_therapy_rate_pct}%` },
            { label: "Hemangioblastoma %", value: `${overview?.hemangioblastoma_rate_pct}%` },
            { label: "Mean Age Dx", value: overview?.mean_age_at_dx },
          ].map(s => (
            <div key={s.label} style={{ background: "#1a1a2e", borderRadius: 6, padding: "10px 12px", textAlign: "center" }}>
              <div style={{ fontSize: 18, fontWeight: 700, color: "#2980b9" }}>{s.value}</div>
              <div style={{ fontSize: 10, color: "#7f8c8d", marginTop: 2 }}>{s.label}</div>
            </div>
          ))}
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 8, marginBottom: 20, flexWrap: "wrap" }}>
          {["overview", "breakdown", "definitions"].map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              padding: "6px 16px", borderRadius: 4, border: "none", cursor: "pointer", fontSize: 12,
              background: tab === t ? "#2980b9" : "#1a1a2e",
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
              <div style={{ fontSize: 13, fontWeight: 700, color: "#2980b9", marginBottom: 10 }}>
                🔑 Key Clinical Facts
              </div>
              {(overview?.key_facts || []).map((fact, i) => (
                <div key={i} style={{ fontSize: 11, color: "#bdc3c7", marginBottom: 6, paddingLeft: 12, borderLeft: "2px solid #2980b9" }}>
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
                      { label: "Radiation CI", value: `${bd.radiation_contraindicated_pct}%` },
                      { label: "Targeted Therapy", value: `${bd.targeted_therapy_pct}%` },
                      { label: "SEGA Risk", value: `${bd.sega_risk_pct}%` },
                      { label: "Hemangioblastoma", value: `${bd.hemangioblastoma_pct}%` },
                      { label: "MB SHH", value: `${bd.medulloblastoma_shh_pct}%` },
                      { label: "Bilateral VS", value: `${bd.bilateral_vs_pct}%` },
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
              <div key={key} style={{ background: "#1a1a2e", borderRadius: 8, padding: 14, marginBottom: 12, borderLeft: "4px solid #2980b9" }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: "#2980b9", marginBottom: 6 }}>
                  {key.replace(/_/g, " ").toUpperCase()}
                </div>
                <div style={{ fontSize: 11, color: "#bdc3c7", lineHeight: 1.6 }}>{text}</div>
              </div>
            ))}

            {/* Key clinical distinctions */}
            <div style={{ background: "#1a1a2e", borderRadius: 8, padding: 14, marginTop: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#2980b9", marginBottom: 10 }}>
                🔑 Key Clinical Distinctions
              </div>
              {(definitions.key_clinical_distinctions || []).map((d, i) => (
                <div key={i} style={{ fontSize: 11, color: "#bdc3c7", marginBottom: 6, paddingLeft: 12, borderLeft: "2px solid #2980b9" }}>
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
