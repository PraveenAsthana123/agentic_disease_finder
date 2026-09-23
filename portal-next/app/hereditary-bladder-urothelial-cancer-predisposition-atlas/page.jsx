"use client";
import { useEffect, useState } from "react";

const GENE_COLORS = {
  MSH2:  "#2980b9",
  MLH1:  "#8e44ad",
  MSH6:  "#16a085",
  BRCA1: "#c0392b",
  BRCA2: "#27ae60",
  RB1:   "#e67e22",
  TP53:  "#d35400",
  PTEN:  "#f39c12",
};

const GENE_INFO = {
  MSH2:  { full: "MSH2 (MutS Homolog 2 / Lynch Type 2 / Muir-Torre PATHOGNOMONIC / Urothelial 10-14% HIGHEST / EPCAM MLPA MANDATORY)", locus: "2p21", size: "934 aa / 100 kDa", inh: "AD LOF", risk: "Urothelial 10-14% HIGHEST Lynch gene; Muir-Torre sebaceous PATHOGNOMONIC; EPCAM MLPA MANDATORY (30% missed); annual cystoscopy MANDATORY; pembrolizumab FDA2017 MSI-H" },
  MLH1:  { full: "MLH1 (MutL Homolog 1 / Lynch Type 1 / MSI-H PATHOGNOMONIC / Pembrolizumab FDA2017 / Constitutional Methylation NOT Inherited)", locus: "3p22.2", size: "756 aa / 85 kDa", inh: "AD LOF", risk: "Urothelial 2-8% MSI-H; IHC MLH1/PMS2 loss PATHOGNOMONIC; pembrolizumab FDA2017 APPROVED; constitutional methylation ~15% NOT heritable" },
  MSH6:  { full: "MSH6 (MutS Homolog 6 / Lynch Type 3 / Endometrial 71% HIGHEST / Urothelial 5-7% / 4 Pseudogenes Sequencing Pitfall)", locus: "2p16.3", size: "1360 aa / 160 kDa", inh: "AD LOF", risk: "Endometrial 71% HIGHEST single MMR gene; urothelial 5-7%; 4 pseudogenes — MLPA mandatory MSH6; pembrolizumab FDA2017 MSI-H" },
  BRCA1: { full: "BRCA1 (BRCT Scaffold / HBOC-1 / HRD Cisplatin Preferred / Olaparib FDA2020 / Urothelial 1.5-2x elevated)", locus: "17q21.31", size: "1863 aa / 210 kDa", inh: "AD LOF", risk: "Urothelial 1.5-2x HRD; CISPLATIN PREFERRED over carboplatin (stronger HRD advantage); olaparib FDA2020 HRD maintenance; breast 72% dominant" },
  BRCA2: { full: "BRCA2 (HR Scaffold / FANCD1 / HBOC-2 / Urothelial 2-3x Elevated / Olaparib FDA2020 HRD / Cisplatin Sensitive)", locus: "13q12.3", size: "3418 aa / 384 kDa", inh: "AD LOF / AR biallelic FA", risk: "Urothelial 2-3x (higher than BRCA1 elevation); olaparib FDA2020 HRD-positive urothelial; cisplatin HRD-sensitive; pancreatic 5-7% → EUS/MRI 50yr+" },
  RB1:   { full: "RB1 (pRb E2F Regulator / Bilateral RB PATHOGNOMONIC / Secondary TCC post-RT 5x / AVOID Radiation / CDK4-6i INACTIVE)", locus: "13q14.2", size: "928 aa / 105 kDa", inh: "AD LOF", risk: "Secondary TCC post-RT 5x; AVOID RADIATION RB1 germline; CDK4-6i INACTIVE RB1-null; annual cystoscopy bilateral RB survivors LIFELONG MANDATORY" },
  TP53:  { full: "TP53 (p53 / Li-Fraumeni Syndrome / AVOID RADIATION ABSOLUTELY / WB-MRI Toronto / Trimodality RT CONTRAINDICATED LFS)", locus: "17p13.1", size: "393 aa / 43 kDa", inh: "AD LOF", risk: "Urothelial 2-3x elevated LFS; AVOID RADIATION ABSOLUTELY; WB-MRI Toronto annual; trimodality RT bladder CONTRAINDICATED LFS; cystectomy mandatory LFS MIBC" },
  PTEN:  { full: "PTEN (Dual Phosphatase / Cowden PHTS / Macrocephaly PATHOGNOMONIC / Lhermitte-Duclos PATHOGNOMONIC / Urothelial 5-8x / mTOR Therapy)", locus: "10q23.31", size: "403 aa / 47 kDa", inh: "AD LOF", risk: "Urothelial 5-8x Cowden PHTS; macrocephaly PATHOGNOMONIC; Lhermitte-Duclos PATHOGNOMONIC; mTOR hyperactivation → everolimus/alpelisib; annual cystoscopy 30-35yr+" },
};

export default function HereditaryBladderUrothelialCancerPredispositionAtlasPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [tab, setTab] = useState("overview");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = "/api/hereditary-bladder-urothelial-cancer-predisposition-atlas";
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-6 text-white">Loading Hereditary Bladder Urothelial Cancer Predisposition Atlas…</div>;
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
            🔬 Hereditary Bladder Urothelial Cancer Predisposition Atlas
          </h1>
          <div style={{ fontSize: 12, color: "#95a5a6", marginTop: 6 }}>
            Complete 8-Gene Reference · MSH2-MLH1-MSH6-BRCA1-BRCA2-RB1-TP53-PTEN ·
            320-Patient Aggregate (8×40, seeds 3446–3453) · Lynch MSI-H / HRD Cisplatin-PARP / RB1 Secondary TCC / LFS Radiation-CI / Cowden mTOR
          </div>
          <div style={{ marginTop: 8, padding: "6px 12px", background: "#1a1a2e", borderRadius: 4, fontSize: 11, color: "#e74c3c", display: "inline-block" }}>
            ⚠ KEY RULES: MSH2 = Urothelial HIGHEST Lynch (10-14%) + EPCAM MLPA MANDATORY | BRCA1 = CISPLATIN preferred (not carboplatin) + olaparib | RB1 = AVOID radiation + CDK4-6i INACTIVE | TP53/LFS = AVOID RADIATION ABSOLUTELY + cystectomy over trimodality RT | PTEN/Cowden = macrocephaly PATHOGNOMONIC + mTOR therapy
          </div>
        </div>

        {/* Cohort Summary */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(7, 1fr)", gap: 10, marginBottom: 20 }}>
          {[
            { label: "Total Patients", value: overview?.total_patients },
            { label: "Genes", value: (overview?.genes||[]).length },
            { label: "MSI-H %", value: overview?.msi_h_rate_pct != null ? `${overview.msi_h_rate_pct}%` : "—" },
            { label: "HRD+ %", value: overview?.hrd_positive_rate_pct != null ? `${overview.hrd_positive_rate_pct}%` : "—" },
            { label: "Pembrolizumab %", value: overview?.pembrolizumab_eligible_rate_pct != null ? `${overview.pembrolizumab_eligible_rate_pct}%` : "—" },
            { label: "Olaparib %", value: overview?.olaparib_eligible_rate_pct != null ? `${overview.olaparib_eligible_rate_pct}%` : "—" },
            { label: "Mean Age Dx", value: overview?.mean_age_at_dx != null ? `${overview.mean_age_at_dx}yr` : "—" },
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

                  <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 8, marginBottom: 12 }}>
                    {[
                      { label: "MSI-H %", value: `${bd.msi_h_pct ?? "—"}%` },
                      { label: "HRD+ %", value: `${bd.hrd_positive_pct ?? "—"}%` },
                      { label: "Pembro %", value: `${bd.pembrolizumab_eligible_pct ?? "—"}%` },
                      { label: "Olaparib %", value: `${bd.olaparib_eligible_pct ?? "—"}%` },
                      { label: "Radiation CI %", value: `${bd.radiation_ci_pct ?? "—"}%` },
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
              <div key={key} style={{ background: "#1a1a2e", borderRadius: 8, padding: 14, marginBottom: 12, borderLeft: "4px solid #2980b9" }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: "#2980b9", marginBottom: 6 }}>
                  {key.replace(/_/g, " ").toUpperCase()}
                </div>
                <div style={{ fontSize: 11, color: "#bdc3c7", lineHeight: 1.6 }}>{text}</div>
              </div>
            ))}

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
