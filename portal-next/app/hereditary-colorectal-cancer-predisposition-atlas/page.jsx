"use client";
import { useEffect, useState } from "react";

const GENE_COLORS = {
  APC:   "#1a6b4a",
  MUTYH: "#c0392b",
  MLH1:  "#2980b9",
  MSH2:  "#8e44ad",
  MSH6:  "#d35400",
  PMS2:  "#16a085",
  STK11: "#f39c12",
  SMAD4: "#27ae60",
};

const GENE_INFO = {
  APC:   { full: "APC (WNT-Gatekeeper / FAP-100% CRC / CHRPE-PATHOGNOMONIC)", locus: "5q22.2", size: "2843 aa / 310 kDa", inh: "AD LOF", risk: "100% CRC lifetime" },
  MUTYH: { full: "MUTYH (DNA-Glycosylase / MAP-Biallelic / Y179C-G396D)", locus: "1p34.1", size: "546 aa / 60 kDa", inh: "AR LOF (biallelic)", risk: "43-75% CRC biallelic" },
  MLH1:  { full: "MLH1 (MutLα-85kDa / Lynch1 / BRAF-V600E-Gate)", locus: "3p22.2", size: "756 aa / 85 kDa", inh: "AD LOF", risk: "40-80% CRC Lynch1" },
  MSH2:  { full: "MSH2 (MutSα+β / Lynch2 / Urothelial-14%-HIGHEST)", locus: "2p21", size: "934 aa / 105 kDa", inh: "AD LOF", risk: "40-80% CRC; 14% urothelial HIGHEST" },
  MSH6:  { full: "MSH6 (MutSα / Lynch3 / Endometrial-71%-HIGHEST)", locus: "2p16.3", size: "1360 aa / 160 kDa", inh: "AD LOF", risk: "10-22% CRC; 71% endometrial HIGHEST" },
  PMS2:  { full: "PMS2 (MutLα-Endonuclease / Lynch4-LOWEST / 4-Pseudogenes)", locus: "7p22.1", size: "862 aa / 96 kDa", inh: "AD LOF / AR biallelic CMMRD", risk: "15-20% CRC LOWEST Lynch" },
  STK11: { full: "STK11 (LKB1-AMPK / PJS / Perioral-Pigmentation-PATHOGNOMONIC)", locus: "19p13.3", size: "433 aa / 48 kDa", inh: "AD LOF", risk: "35-40% CRC; pancreatic 36% HIGHEST" },
  SMAD4: { full: "SMAD4 (TGF-β-Mediator / JPS-HHT-Overlap / Aortic-Dilatation)", locus: "18q21.2", size: "552 aa / 60 kDa", inh: "AD LOF", risk: "40-70% CRC JPS; aortic dilation 25-30%" },
};

export default function HereditoryCRCPredispositionAtlasPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [tab, setTab] = useState("overview");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = "/api/hereditary-colorectal-cancer-predisposition-atlas";
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-6 text-white">Loading Hereditary Colorectal Cancer Predisposition Atlas…</div>;
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
            🏥 Hereditary Colorectal Cancer Predisposition Atlas
          </h1>
          <div style={{ fontSize: 12, color: "#95a5a6", marginTop: 6 }}>
            Complete 8-Gene Reference · APC-MUTYH-MLH1-MSH2-MSH6-PMS2-STK11-SMAD4 ·
            320-Patient Aggregate (8×40, seeds 3398-3405) · FAP / MAP / Lynch1-4 / PJS / JPS
          </div>
          <div style={{ marginTop: 8, padding: "6px 12px", background: "#1a1a2e", borderRadius: 4, fontSize: 11, color: "#e74c3c", display: "inline-block" }}>
            ⚠ KEY RULES: APC = 100% CRC — colectomy MANDATORY | MUTYH MAP = biallelic ONLY (MSS not MSI-H) | MLH1 BRAF-V600E gate | MSH2 urothelial 14% HIGHEST | MSH6 = endometrial 71% DOMINATES | STK11 GI from 8yr | SMAD4 HHT = SMAD4-ONLY
          </div>
        </div>

        {/* Cohort Summary */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 10, marginBottom: 20 }}>
          {[
            { label: "Total Patients", value: overview?.total_patients },
            { label: "Genes", value: (overview?.genes||[]).length },
            { label: "MSI-H Rate %", value: `${overview?.msi_h_rate_pct}%` },
            { label: "Immunotherapy Eligible %", value: `${overview?.immunotherapy_eligible_pct}%` },
            { label: "Prophylactic Procedure %", value: `${overview?.prophylactic_procedure_pct}%` },
            { label: "Mean Age Dx", value: overview?.mean_age_at_dx },
          ].map(s => (
            <div key={s.label} style={{ background: "#1a1a2e", borderRadius: 6, padding: "10px 12px", textAlign: "center" }}>
              <div style={{ fontSize: 18, fontWeight: 700, color: "#1a6b4a" }}>{s.value ?? "—"}</div>
              <div style={{ fontSize: 10, color: "#95a5a6", marginTop: 2 }}>{s.label}</div>
            </div>
          ))}
        </div>

        {/* Seed/Stats row */}
        <div style={{ marginBottom: 16, padding: "8px 12px", background: "#1a1a2e", borderRadius: 6, fontSize: 11, display: "flex", gap: 24, flexWrap: "wrap" }}>
          <span style={{ color: "#e74c3c", fontWeight: 700 }}>Metastatic: {overview?.metastatic_pct}%</span>
          <span style={{ color: "#95a5a6" }}>Seed range: {overview?.seed_range}</span>
          <span style={{ color: "#f39c12" }}>Aspirin CAPP2 prescribed: {overview?.aspirin_prescribed_pct}%</span>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 8, marginBottom: 20, borderBottom: "1px solid #2c2c44" }}>
          {["overview", "gene-table", "clinical-atlas", "definitions"].map(t => (
            <button key={t} onClick={() => setTab(t)}
              style={{
                padding: "8px 16px", border: "none", borderRadius: "4px 4px 0 0",
                background: tab === t ? "#1a6b4a" : "#1a1a2e",
                color: tab === t ? "#fff" : "#95a5a6",
                cursor: "pointer", fontSize: 12, textTransform: "capitalize",
              }}>
              {t.replace("-", " ")}
            </button>
          ))}
        </div>

        {/* TAB: Overview */}
        {tab === "overview" && (
          <div>
            {/* Gene colour legend */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 20 }}>
              {genes.map(gene => (
                <div key={gene} style={{ display: "flex", alignItems: "center", gap: 6, background: "#1a1a2e", borderRadius: 4, padding: "6px 10px" }}>
                  <div style={{ width: 10, height: 10, borderRadius: "50%", background: GENE_COLORS[gene] || "#666" }} />
                  <span style={{ fontSize: 11, fontWeight: 700, color: GENE_COLORS[gene] || "#fff" }}>{gene}</span>
                  <span style={{ fontSize: 10, color: "#7f8c8d" }}>{GENE_INFO[gene]?.risk}</span>
                </div>
              ))}
            </div>

            {/* Key facts */}
            <div style={{ background: "#1a1a2e", borderRadius: 8, padding: 16, marginBottom: 20 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#1a6b4a", marginBottom: 12 }}>Key Clinical Facts</div>
              {(overview?.key_facts || []).map((f, i) => (
                <div key={i} style={{ fontSize: 11, color: "#b0b0b0", marginBottom: 6, paddingLeft: 12, borderLeft: "3px solid " + (GENE_COLORS[f.split(":")[0]] || "#1a6b4a") }}>
                  {f}
                </div>
              ))}
            </div>

            {/* Per-gene cohort counts */}
            <div style={{ background: "#1a1a2e", borderRadius: 8, padding: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#1a6b4a", marginBottom: 12 }}>Cohort Distribution (n={overview?.total_patients})</div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
                {genes.map(gene => {
                  const info = GENE_INFO[gene] || {};
                  const n = overview?.gene_counts?.[gene] ?? 0;
                  return (
                    <div key={gene} style={{ background: "#0f1117", borderRadius: 6, padding: 10, borderLeft: "3px solid " + (GENE_COLORS[gene] || "#666") }}>
                      <div style={{ fontWeight: 700, color: GENE_COLORS[gene] || "#fff", fontSize: 14 }}>{gene}</div>
                      <div style={{ fontSize: 11, color: "#95a5a6" }}>{info.locus} · {info.size}</div>
                      <div style={{ fontSize: 10, color: "#7f8c8d" }}>{info.inh}</div>
                      <div style={{ fontSize: 10, color: "#e67e22", marginTop: 4 }}>{info.risk}</div>
                      <div style={{ fontSize: 12, fontWeight: 700, color: "#ecf0f1", marginTop: 4 }}>n = {n}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* TAB: Gene Table */}
        {tab === "gene-table" && (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr style={{ background: "#1a1a2e", color: "#95a5a6" }}>
                  {["Gene", "Locus", "Syndrome", "CRC Risk", "Key Distinction", "MSI Status", "Surveillance Start", "Targeted Rx"].map(h => (
                    <th key={h} style={{ padding: "8px 10px", textAlign: "left", borderBottom: "1px solid #2c2c44" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {genes.map((gene, idx) => {
                  const bd = breakdown?.breakdown?.[gene];
                  const info = GENE_INFO[gene] || {};
                  const gi = bd?.gene_info || {};
                  return (
                    <tr key={gene} style={{ background: idx % 2 === 0 ? "#0f1117" : "#13141f", borderBottom: "1px solid #1a1a2e" }}>
                      <td style={{ padding: "8px 10px", fontWeight: 700, color: GENE_COLORS[gene] || "#fff" }}>{gene}</td>
                      <td style={{ padding: "8px 10px", color: "#95a5a6" }}>{info.locus}</td>
                      <td style={{ padding: "8px 10px", color: "#b0b0b0", maxWidth: 160 }}>{gi.syndrome?.split(" —")[0] || info.full?.split(" /")[0]}</td>
                      <td style={{ padding: "8px 10px", color: "#e67e22", fontWeight: 700 }}>{gi.crc_risk?.split(";")[0]}</td>
                      <td style={{ padding: "8px 10px", color: "#f39c12", fontSize: 10, maxWidth: 180 }}>{gi.pathognomonic?.substring(0, 100)}…</td>
                      <td style={{ padding: "8px 10px", color: gene === "MUTYH" ? "#e74c3c" : "#3498db" }}>
                        {gene === "MUTYH" ? "MSS (NOT MSI-H)" : gene === "MSH6" ? "MSI-L 30% FN" : gene === "APC" || gene === "STK11" || gene === "SMAD4" ? "MSS" : "MSI-H typical"}
                      </td>
                      <td style={{ padding: "8px 10px", color: "#27ae60", fontSize: 10 }}>
                        {gene === "APC" ? "12-14yr" : gene === "MUTYH" ? "18-25yr (biallelic)" : gene === "MLH1" || gene === "MSH2" ? "25yr" : gene === "MSH6" ? "30-35yr" : gene === "PMS2" ? "35yr" : gene === "STK11" ? "8yr GI" : gene === "SMAD4" ? "15yr + echo" : "25yr"}
                      </td>
                      <td style={{ padding: "8px 10px", color: "#9b59b6", fontSize: 10, maxWidth: 160 }}>
                        {gene === "APC" ? "Sulindac/celecoxib; colectomy" : gene === "MLH1" || gene === "MSH2" || gene === "MSH6" || gene === "PMS2" ? "Aspirin CAPP2 + Pembrolizumab MSI-H" : gene === "STK11" ? "mTOR investigational" : gene === "SMAD4" ? "Aortic surgery if >50mm" : "Polypectomy; standard chemo"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* TAB: Clinical Atlas */}
        {tab === "clinical-atlas" && (
          <div>
            {genes.map(gene => {
              const bd = breakdown?.breakdown?.[gene];
              if (!bd) return null;
              const gi = bd.gene_info || {};
              return (
                <div key={gene} style={{ background: "#1a1a2e", borderRadius: 8, padding: 16, marginBottom: 16, borderLeft: "4px solid " + (GENE_COLORS[gene] || "#666") }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                    <div>
                      <span style={{ fontSize: 16, fontWeight: 700, color: GENE_COLORS[gene] || "#fff" }}>{gene}</span>
                      <span style={{ fontSize: 11, color: "#95a5a6", marginLeft: 12 }}>{GENE_INFO[gene]?.locus} · {GENE_INFO[gene]?.size} · {gi.inheritance}</span>
                    </div>
                    <div style={{ fontSize: 11, color: "#e67e22", fontWeight: 700 }}>{gi.crc_risk?.split(";")[0]}</div>
                  </div>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 8, marginBottom: 10 }}>
                    {[
                      { label: "MSI-H %", val: `${bd.msi_h_pct}%` },
                      { label: "Immunotherapy %", val: `${bd.immunotherapy_pct}%` },
                      { label: "Prophylactic %", val: `${bd.prophylactic_pct}%` },
                      { label: "Aspirin %", val: `${bd.aspirin_pct}%` },
                    ].map(s => (
                      <div key={s.label} style={{ background: "#0f1117", borderRadius: 4, padding: "8px 10px", textAlign: "center" }}>
                        <div style={{ fontSize: 14, fontWeight: 700, color: GENE_COLORS[gene] || "#fff" }}>{s.val}</div>
                        <div style={{ fontSize: 9, color: "#95a5a6" }}>{s.label}</div>
                      </div>
                    ))}
                  </div>

                  <div style={{ fontSize: 10, color: "#e74c3c", marginBottom: 8, background: "#1a0a0a", padding: "6px 10px", borderRadius: 4 }}>
                    <strong>KEY RULE:</strong> {gi.key_rule}
                  </div>
                  <div style={{ fontSize: 10, color: "#f39c12", marginBottom: 8 }}>
                    <strong>AVOID:</strong> {gi.key_avoid}
                  </div>
                  <div style={{ fontSize: 10, color: "#27ae60", marginBottom: 8 }}>
                    <strong>PATHOGNOMONIC:</strong> {gi.pathognomonic}
                  </div>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 8 }}>
                    <div>
                      <div style={{ fontSize: 11, color: "#3498db", fontWeight: 700, marginBottom: 4 }}>Surveillance</div>
                      {(gi.surveillance || []).slice(0, 4).map((s, i) => (
                        <div key={i} style={{ fontSize: 10, color: "#95a5a6", marginBottom: 2 }}>• {s}</div>
                      ))}
                    </div>
                    <div>
                      <div style={{ fontSize: 11, color: "#9b59b6", fontWeight: 700, marginBottom: 4 }}>Treatment</div>
                      {(bd.treatment_protocols || []).slice(0, 4).map((t, i) => (
                        <div key={i} style={{ fontSize: 10, color: "#95a5a6", marginBottom: 2 }}>• {t}</div>
                      ))}
                    </div>
                  </div>

                  {bd.top_variants?.length > 0 && (
                    <div style={{ marginTop: 10, fontSize: 10, color: "#7f8c8d" }}>
                      <strong style={{ color: "#95a5a6" }}>Top variants: </strong>
                      {bd.top_variants.map(v => `${v.variant} (n=${v.count})`).join(" · ")}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* TAB: Definitions */}
        {tab === "definitions" && (
          <div>
            <div style={{ marginBottom: 16, padding: "8px 12px", background: "#1a1a2e", borderRadius: 6, fontSize: 11, color: "#e74c3c" }}>
              ⚠ Key clinical distinctions for hereditary CRC management
            </div>
            {(definitions?.key_clinical_distinctions || []).map((d, i) => (
              <div key={i} style={{ fontSize: 11, color: "#b0b0b0", marginBottom: 8, paddingLeft: 12, borderLeft: "3px solid " + (GENE_COLORS[d.split(":")[0]] || "#1a6b4a") }}>
                {d}
              </div>
            ))}
            <div style={{ marginTop: 20 }}>
              {Object.entries(definitions?.definitions || {}).map(([key, val]) => (
                <div key={key} style={{ background: "#1a1a2e", borderRadius: 6, padding: 14, marginBottom: 12 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: "#1a6b4a", marginBottom: 6, textTransform: "uppercase" }}>
                    {key.replace(/_/g, " ")}
                  </div>
                  <div style={{ fontSize: 11, color: "#b0b0b0", lineHeight: 1.7 }}>{val}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
