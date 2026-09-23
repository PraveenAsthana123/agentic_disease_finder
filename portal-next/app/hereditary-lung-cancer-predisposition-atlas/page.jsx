"use client";
import { useEffect, useState } from "react";

const GENE_COLORS = {
  EGFR:  "#2980b9",
  BRCA2: "#8e44ad",
  TP53:  "#c0392b",
  STK11: "#27ae60",
  DICER1:"#d35400",
  BAP1:  "#16a085",
  FLCN:  "#f39c12",
  NF1:   "#1a6b4a",
};

const GENE_INFO = {
  EGFR:  { full: "EGFR (Germline RTK GOF / Familial NSCLC / Osimertinib Eligible)", locus: "7p11.2", size: "1210 aa / 134 kDa", inh: "AD GOF germline", risk: "Familial NSCLC <1%; lung adenocarcinoma; germline T790M osimertinib ONLY" },
  BRCA2: { full: "BRCA2 (FANCD1 HR Mediator / HBOC2 / HRD PARPi Eligible)", locus: "13q12.3", size: "3418 aa / 384 kDa", inh: "AD LOF", risk: "Lung adenocarcinoma 2-3x; HRD cisplatin-sensitive; PARPi eligible" },
  TP53:  { full: "TP53 (p53 Tumour Suppressor / Li-Fraumeni / AVOID RADIATION)", locus: "17p13.1", size: "393 aa / 43 kDa", inh: "AD LOF", risk: "Lung 2-5x elevated; AVOID RADIATION ABSOLUTELY; WB-MRI Toronto MANDATORY" },
  STK11: { full: "STK11 (LKB1 AMPK Kinase / Peutz-Jeghers / NSCLC 16x RR HIGHEST)", locus: "19p13.3", size: "433 aa / 48 kDa", inh: "AD LOF", risk: "NSCLC 16x RR HIGHEST; STK11+KRAS = immunotherapy-cold COLD TUMOUR" },
  DICER1:{ full: "DICER1 (RNase-III Endoribonuclease / DICER1 Syndrome / PPB PATHOGNOMONIC)", locus: "14q32.13", size: "1922 aa / 218 kDa", inh: "AD LOF", risk: "PPB type I/II/III PATHOGNOMONIC childhood; pulmonary blastoma adult" },
  BAP1:  { full: "BAP1 (Deubiquitinase / BAP1-TPDS / Mesothelioma 8-10% HIGHEST)", locus: "3p21.1", size: "729 aa / 80 kDa", inh: "AD LOF", risk: "Mesothelioma 8-10% HIGHEST; asbestos synergy MANDATORY avoided; uveal 50%" },
  FLCN:  { full: "FLCN (Folliculin / Birt-Hogg-Dubé / Pneumothorax 40% HIGHEST)", locus: "17p11.2", size: "579 aa / 64 kDa", inh: "AD LOF", risk: "Pulmonary cysts PATHOGNOMONIC; pneumothorax 40% HIGHEST hereditary" },
  NF1:   { full: "NF1 (Neurofibromin RAS-GAP / NF1 / MPNST SARCOMA not NSCLC)", locus: "17q11.2", size: "2839 aa / 319 kDa", inh: "AD LOF", risk: "NSCLC 2-3x; MPNST 8-13% = SARCOMA (doxorubicin), NOT platinum" },
};

export default function HereditaryLungCancerPredispositionAtlasPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [tab, setTab] = useState("overview");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = "/api/hereditary-lung-cancer-predisposition-atlas";
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-6 text-white">Loading Hereditary Lung Cancer Predisposition Atlas…</div>;
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
            🫁 Hereditary Lung Cancer Predisposition Atlas
          </h1>
          <div style={{ fontSize: 12, color: "#95a5a6", marginTop: 6 }}>
            Complete 8-Gene Reference · EGFR-BRCA2-TP53-STK11-DICER1-BAP1-FLCN-NF1 ·
            320-Patient Aggregate (8×40, seeds 3414-3421) · Familial NSCLC / LFS / PJS / DICER1 / BAP1-TPDS / BHD / NF1
          </div>
          <div style={{ marginTop: 8, padding: "6px 12px", background: "#1a1a2e", borderRadius: 4, fontSize: 11, color: "#e74c3c", display: "inline-block" }}>
            ⚠ KEY RULES: EGFR germline T790M = osimertinib ONLY | TP53/LFS = AVOID RADIATION ABSOLUTELY | STK11+KRAS = immunotherapy-COLD | DICER1 = annual CT birth-to-8 MANDATORY | BAP1 = asbestos MANDATORY avoided | NF1 MPNST = sarcoma NOT lung cancer (doxorubicin)
          </div>
        </div>

        {/* Cohort Summary */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 10, marginBottom: 20 }}>
          {[
            { label: "Total Patients", value: overview?.total_patients },
            { label: "Genes", value: (overview?.genes||[]).length },
            { label: "EGFR-TKI Eligible %", value: `${overview?.egfr_tki_eligible_rate_pct}%` },
            { label: "PARPi Eligible %", value: `${overview?.parpi_eligible_rate_pct}%` },
            { label: "Pneumothorax %", value: `${overview?.pneumothorax_rate_pct}%` },
            { label: "Mean Age Dx", value: overview?.mean_age_at_dx },
          ].map(s => (
            <div key={s.label} style={{ background: "#1a1a2e", borderRadius: 6, padding: "10px 12px", textAlign: "center" }}>
              <div style={{ fontSize: 18, fontWeight: 700, color: "#2980b9" }}>{s.value}</div>
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
                background: tab === t ? "#2980b9" : "#1a1a2e", color: tab === t ? "#fff" : "#95a5a6" }}>
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        {tab === "overview" && overview && (
          <div>
            <div style={{ background: "#1a1a2e", borderRadius: 6, padding: 16, marginBottom: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#2980b9", marginBottom: 10 }}>Seed Range: {overview.seed_range}</div>
              <div style={{ fontSize: 12, color: "#95a5a6", marginBottom: 8 }}>Key Clinical Facts:</div>
              {(overview.key_facts || []).map((f, i) => (
                <div key={i} style={{ fontSize: 11, color: "#bdc3c7", marginBottom: 6, paddingLeft: 10, borderLeft: "2px solid #2980b9" }}>
                  {f}
                </div>
              ))}
            </div>
            <div style={{ background: "#1a1a2e", borderRadius: 6, padding: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#2980b9", marginBottom: 10 }}>Gene Distribution (n=40 per gene)</div>
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
                      { label: "TKI/PARPi Eligible %", value: gene === "EGFR" ? `${gd.egfr_tki_eligible_pct}%` : `${gd.parpi_eligible_pct}%` },
                      { label: "Pneumothorax %", value: `${gd.pneumothorax_pct}%` },
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
              <div style={{ fontSize: 13, fontWeight: 700, color: "#2980b9", marginBottom: 12 }}>Key Clinical Distinctions</div>
              {(definitions.key_clinical_distinctions || []).map((d, i) => (
                <div key={i} style={{ fontSize: 11, color: "#bdc3c7", marginBottom: 8, paddingLeft: 10, borderLeft: "2px solid #2980b9" }}>
                  {d}
                </div>
              ))}
            </div>
            {Object.entries(definitions.definitions || {}).map(([key, val]) => (
              <div key={key} style={{ background: "#1a1a2e", borderRadius: 6, padding: 14, marginBottom: 10 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: "#2980b9", marginBottom: 6, textTransform: "uppercase" }}>{key.replace(/_/g, " ")}</div>
                <div style={{ fontSize: 11, color: "#bdc3c7", lineHeight: 1.6 }}>{val}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
