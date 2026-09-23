"use client";
import { useEffect, useState } from "react";

const GENE_COLORS = {
  BRCA2:  "#e67e22",
  BRCA1:  "#f39c12",
  ATM:    "#2980b9",
  PALB2:  "#8e44ad",
  MLH1:   "#16a085",
  MSH2:   "#1abc9c",
  HOXB13: "#e74c3c",
  CHEK2:  "#7f8c8d",
};

const GENE_INFO = {
  BRCA2:  { full: "BRCA2 / FANCD1 (HR Mediator, RAD51 Loader)",           locus: "13q12.3",  size: "3418 aa / 384 kDa", inh: "AD LOF" },
  BRCA1:  { full: "BRCA1 / FANCS (HR Scaffold, RING-BRCT)",               locus: "17q21.31", size: "1863 aa / 208 kDa", inh: "AD LOF" },
  ATM:    { full: "ATM (PI3K-like DSB Kinase; HIGH GRADE PCa)",            locus: "11q22.3",  size: "3056 aa / 350 kDa", inh: "AD LOF" },
  PALB2:  { full: "PALB2 / FANCN (BRCA2 Bridge, WD40 Scaffold)",          locus: "16p12.2",  size: "1186 aa / 131 kDa", inh: "AD LOF" },
  MLH1:   { full: "MLH1 (MMR MutL Homologue 1, Lynch Type 1)",            locus: "3p22.2",   size: "756 aa / 85 kDa",   inh: "AD LOF" },
  MSH2:   { full: "MSH2 (MutSalpha/MutSbeta; Lynch Type 2; HIGHEST PCa)", locus: "2p21",     size: "936 aa / 105 kDa",  inh: "AD LOF" },
  HOXB13: { full: "HOXB13 (AR Coregulator; G84E Founder; Prostate-Specific)", locus: "17q21.2", size: "283 aa / 31 kDa", inh: "AD LOF" },
  CHEK2:  { full: "CHEK2 (ATM Effector; I157T / 1100delC Founders)",      locus: "22q12.1",  size: "543 aa / 60 kDa",   inh: "AD LOF" },
};

const SLUG = "hereditary-prostate-cancer-predisposition-atlas";

export default function HeredProstateCancerPredispositionAtlasPage() {
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

  if (loading) return <div className="p-6 text-white">Loading Hereditary Prostate Cancer Predisposition Atlas…</div>;
  if (error)   return <div className="p-6 text-red-400">Error: {error}</div>;

  const geneSummary  = overview?.gene_summary  || [];
  const hierarchy    = overview?.prostate_risk_hierarchy || {};
  const keyRules     = overview?.key_clinical_rules || [];
  const profound     = overview?.profound_trial_summary || {};
  const perGene      = breakdown?.per_gene      || [];
  const profound_a   = breakdown?.profound_trial_cohort_a || {};
  const profound_b   = breakdown?.profound_trial_cohort_b || {};
  const msiVsHrd     = breakdown?.msi_h_vs_hrd_treatment  || {};
  const hoxb13Rules  = breakdown?.hoxb13_g84e_key_rules    || {};
  const defGenes     = definitions?.genes                  || [];
  const keyConcepts  = definitions?.key_concepts           || {};
  const atlasMetadata = definitions?.atlas_metadata        || {};

  const tabs = ["overview", "breakdown", "profound", "definitions"];

  return (
    <div style={{ background: "#0f172a", minHeight: "100vh", color: "#e2e8f0", fontFamily: "monospace" }}>
      <div style={{ background: "#1e293b", borderBottom: "2px solid #334155", padding: "1.5rem 2rem" }}>
        <h1 style={{ fontSize: "1.4rem", fontWeight: 700, color: "#f8fafc", marginBottom: "0.25rem" }}>
          &#x1f9ec; Hereditary Prostate Cancer Predisposition Atlas
        </h1>
        <div style={{ color: "#94a3b8", fontSize: "0.8rem" }}>
          Complete 8-Gene BRCA2-BRCA1-ATM-PALB2-MLH1-MSH2-HOXB13-CHEK2 Reference &nbsp;·&nbsp;
          {overview?.total_patients} patients &nbsp;·&nbsp; seeds {overview?.seeds}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: "0.5rem", padding: "1rem 2rem", borderBottom: "1px solid #334155" }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)}
            style={{
              padding: "0.4rem 1rem", borderRadius: "4px", cursor: "pointer", fontSize: "0.8rem",
              background: tab === t ? "#3b82f6" : "#1e293b",
              color: tab === t ? "#fff" : "#94a3b8",
              border: tab === t ? "none" : "1px solid #334155",
            }}>
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      <div style={{ padding: "1.5rem 2rem" }}>

        {/* ── OVERVIEW ── */}
        {tab === "overview" && (
          <div>
            {/* KPI bar */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(140px,1fr))", gap: "0.75rem", marginBottom: "1.5rem" }}>
              {[
                { label: "Total Patients", value: overview?.total_patients },
                { label: "PCa Cases", value: `${overview?.pca_cases} (${overview?.pca_rate_pct}%)` },
                { label: "High Grade", value: overview?.high_grade_cases },
                { label: "Metastatic", value: overview?.metastatic_cases },
                { label: "HRD Cases", value: `${overview?.hrd_cases} (${overview?.hrd_rate_pct}%)` },
                { label: "MSI-H Cases", value: overview?.msi_h_cases },
                { label: "Early Onset", value: overview?.early_onset_cases },
                { label: "RT Sensitive", value: overview?.rt_sensitivity_cases },
              ].map(k => (
                <div key={k.label} style={{ background: "#1e293b", border: "1px solid #334155", borderRadius: "6px", padding: "0.75rem", textAlign: "center" }}>
                  <div style={{ fontSize: "1.3rem", fontWeight: 700, color: "#60a5fa" }}>{k.value ?? "—"}</div>
                  <div style={{ fontSize: "0.7rem", color: "#94a3b8" }}>{k.label}</div>
                </div>
              ))}
            </div>

            {/* Gene summary table */}
            <h2 style={{ color: "#93c5fd", marginBottom: "0.75rem", fontSize: "1rem" }}>Gene Cohort Summary</h2>
            <div style={{ overflowX: "auto", marginBottom: "1.5rem" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.78rem" }}>
                <thead>
                  <tr style={{ background: "#1e293b" }}>
                    {["Gene", "N", "PCa %", "High Grade", "Metastatic", "HRD %", "MSI-H", "Mean Age"].map(h => (
                      <th key={h} style={{ padding: "0.5rem", borderBottom: "1px solid #334155", color: "#94a3b8", textAlign: "left" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {geneSummary.map(g => (
                    <tr key={g.gene} style={{ borderBottom: "1px solid #1e293b" }}>
                      <td style={{ padding: "0.4rem 0.5rem", color: GENE_COLORS[g.gene] || "#e2e8f0", fontWeight: 700 }}>{g.gene}</td>
                      <td style={{ padding: "0.4rem 0.5rem" }}>{g.n}</td>
                      <td style={{ padding: "0.4rem 0.5rem" }}>{g.pca_pct}%</td>
                      <td style={{ padding: "0.4rem 0.5rem" }}>{g.high_grade}</td>
                      <td style={{ padding: "0.4rem 0.5rem" }}>{g.metastatic}</td>
                      <td style={{ padding: "0.4rem 0.5rem" }}>{g.hrd_pct}%</td>
                      <td style={{ padding: "0.4rem 0.5rem" }}>{g.msi_h}</td>
                      <td style={{ padding: "0.4rem 0.5rem" }}>{g.mean_age}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Risk hierarchy */}
            <h2 style={{ color: "#93c5fd", marginBottom: "0.75rem", fontSize: "1rem" }}>Prostate Cancer Risk Hierarchy</h2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: "0.4rem", marginBottom: "1.5rem" }}>
              {Object.entries(hierarchy).map(([gene, risk]) => (
                <><div key={gene + "k"} style={{ background: "#1e293b", padding: "0.4rem 0.7rem", borderRadius: "4px", color: GENE_COLORS[gene.split("_")[0]] || "#f59e0b", fontWeight: 700, fontSize: "0.78rem" }}>{gene}</div>
                  <div key={gene + "v"} style={{ background: "#0f172a", padding: "0.4rem 0.7rem", borderRadius: "4px", fontSize: "0.78rem", color: "#e2e8f0" }}>{risk}</div></>
              ))}
            </div>

            {/* Key clinical rules */}
            <h2 style={{ color: "#93c5fd", marginBottom: "0.75rem", fontSize: "1rem" }}>Key Clinical Rules</h2>
            <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
              {keyRules.map((r, i) => (
                <li key={i} style={{ background: "#1e293b", border: "1px solid #334155", borderRadius: "6px", padding: "0.75rem", marginBottom: "0.5rem", fontSize: "0.8rem", lineHeight: 1.5 }}>
                  <span style={{ color: "#f59e0b", marginRight: "0.4rem" }}>&#9658;</span>{r}
                </li>
              ))}
            </ul>

            {/* PROfound trial summary */}
            {Object.keys(profound).length > 0 && (
              <>
                <h2 style={{ color: "#93c5fd", marginTop: "1.5rem", marginBottom: "0.75rem", fontSize: "1rem" }}>PROfound Trial Summary</h2>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: "0.4rem" }}>
                  {Object.entries(profound).map(([k, v]) => (
                    <><div key={k + "k"} style={{ background: "#1e293b", padding: "0.4rem 0.7rem", borderRadius: "4px", color: "#60a5fa", fontSize: "0.78rem" }}>{k}</div>
                      <div key={k + "v"} style={{ background: "#0f172a", padding: "0.4rem 0.7rem", borderRadius: "4px", fontSize: "0.78rem", color: "#e2e8f0" }}>{String(v)}</div></>
                  ))}
                </div>
              </>
            )}
          </div>
        )}

        {/* ── BREAKDOWN ── */}
        {tab === "breakdown" && (
          <div>
            {perGene.map(g => (
              <div key={g.gene} style={{ background: "#1e293b", border: `1px solid ${GENE_COLORS[g.gene] || "#334155"}`, borderRadius: "8px", padding: "1rem", marginBottom: "1rem" }}>
                <h3 style={{ color: GENE_COLORS[g.gene] || "#e2e8f0", marginBottom: "0.5rem" }}>
                  {g.gene} — {g.syndrome}
                </h3>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(110px,1fr))", gap: "0.5rem", marginBottom: "0.75rem" }}>
                  {[
                    { label: "N", value: g.n },
                    { label: "PCa %", value: `${g.pca_pct}%` },
                    { label: "High Grade", value: g.high_grade_n },
                    { label: "Metastatic", value: g.metastatic_n },
                    { label: "HRD %", value: `${g.hrd_pct}%` },
                    { label: "MSI-H", value: g.msi_h_n },
                    { label: "Early Onset", value: g.early_onset_n },
                    { label: "Mean Age", value: g.mean_age },
                  ].map(k => (
                    <div key={k.label} style={{ background: "#0f172a", borderRadius: "4px", padding: "0.5rem", textAlign: "center" }}>
                      <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "#60a5fa" }}>{k.value}</div>
                      <div style={{ fontSize: "0.65rem", color: "#64748b" }}>{k.label}</div>
                    </div>
                  ))}
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem" }}>
                  <div style={{ background: "#0f172a", borderRadius: "4px", padding: "0.6rem" }}>
                    <div style={{ color: "#f87171", fontSize: "0.72rem", marginBottom: "0.3rem", fontWeight: 700 }}>&#9888; AVOID</div>
                    <div style={{ fontSize: "0.78rem", lineHeight: 1.5 }}>{g.key_avoid}</div>
                  </div>
                  <div style={{ background: "#0f172a", borderRadius: "4px", padding: "0.6rem" }}>
                    <div style={{ color: "#4ade80", fontSize: "0.72rem", marginBottom: "0.3rem", fontWeight: 700 }}>&#10003; KEY RULE</div>
                    <div style={{ fontSize: "0.78rem", lineHeight: 1.5 }}>{g.key_rule}</div>
                  </div>
                </div>
              </div>
            ))}

            {/* HOXB13 G84E rules */}
            <h2 style={{ color: "#93c5fd", marginTop: "1.5rem", marginBottom: "0.75rem", fontSize: "1rem" }}>HOXB13 G84E — Prostate-Specific Rules</h2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: "0.4rem", marginBottom: "1.5rem" }}>
              {Object.entries(hoxb13Rules).map(([k, v]) => (
                <><div key={k + "k"} style={{ background: "#1e293b", padding: "0.4rem 0.7rem", borderRadius: "4px", color: "#e74c3c", fontSize: "0.78rem" }}>{k}</div>
                  <div key={k + "v"} style={{ background: "#0f172a", padding: "0.4rem 0.7rem", borderRadius: "4px", fontSize: "0.78rem", color: "#e2e8f0" }}>{String(v)}</div></>
              ))}
            </div>

            {/* MSI vs HRD */}
            <h2 style={{ color: "#93c5fd", marginBottom: "0.75rem", fontSize: "1rem" }}>MSI-H vs HRD Treatment Strategies</h2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: "0.4rem" }}>
              {Object.entries(msiVsHrd).map(([k, v]) => (
                <><div key={k + "k"} style={{ background: "#1e293b", padding: "0.4rem 0.7rem", borderRadius: "4px", color: "#94a3b8", fontSize: "0.78rem" }}>{k}</div>
                  <div key={k + "v"} style={{ background: "#0f172a", padding: "0.4rem 0.7rem", borderRadius: "4px", fontSize: "0.78rem", color: "#e2e8f0" }}>{String(v)}</div></>
              ))}
            </div>
          </div>
        )}

        {/* ── PROFOUND ── */}
        {tab === "profound" && (
          <div>
            <h2 style={{ color: "#93c5fd", marginBottom: "0.75rem", fontSize: "1rem" }}>PROfound Trial — Cohort A (BRCA1/BRCA2)</h2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: "0.4rem", marginBottom: "1.5rem" }}>
              {Object.entries(profound_a).map(([k, v]) => (
                <><div key={k + "k"} style={{ background: "#1e293b", padding: "0.4rem 0.7rem", borderRadius: "4px", color: "#f59e0b", fontSize: "0.78rem" }}>{k}</div>
                  <div key={k + "v"} style={{ background: "#0f172a", padding: "0.4rem 0.7rem", borderRadius: "4px", fontSize: "0.78rem", color: "#e2e8f0" }}>{String(v)}</div></>
              ))}
            </div>

            <h2 style={{ color: "#93c5fd", marginBottom: "0.75rem", fontSize: "1rem" }}>PROfound Trial — Cohort B (ATM + Others)</h2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: "0.4rem", marginBottom: "1.5rem" }}>
              {Object.entries(profound_b).map(([k, v]) => (
                <><div key={k + "k"} style={{ background: "#1e293b", padding: "0.4rem 0.7rem", borderRadius: "4px", color: "#2980b9", fontSize: "0.78rem" }}>{k}</div>
                  <div key={k + "v"} style={{ background: "#0f172a", padding: "0.4rem 0.7rem", borderRadius: "4px", fontSize: "0.78rem", color: "#e2e8f0" }}>{String(v)}</div></>
              ))}
            </div>

            {/* Atlas metadata */}
            <h2 style={{ color: "#93c5fd", marginBottom: "0.75rem", fontSize: "1rem" }}>Atlas Metadata</h2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: "0.4rem" }}>
              {Object.entries(atlasMetadata).map(([k, v]) => (
                <><div key={k + "k"} style={{ background: "#1e293b", padding: "0.4rem 0.7rem", borderRadius: "4px", color: "#94a3b8", fontSize: "0.78rem" }}>{k}</div>
                  <div key={k + "v"} style={{ background: "#0f172a", padding: "0.4rem 0.7rem", borderRadius: "4px", fontSize: "0.78rem", color: "#e2e8f0" }}>{Array.isArray(v) ? v.join(", ") : String(v)}</div></>
              ))}
            </div>
          </div>
        )}

        {/* ── DEFINITIONS ── */}
        {tab === "definitions" && (
          <div>
            {defGenes.map(g => (
              <div key={g.gene} style={{ background: "#1e293b", border: `1px solid ${GENE_COLORS[g.gene] || "#334155"}`, borderRadius: "8px", padding: "1rem", marginBottom: "1rem" }}>
                <h3 style={{ color: GENE_COLORS[g.gene] || "#e2e8f0", marginBottom: "0.25rem" }}>{g.gene} — {g.full_name}</h3>
                <div style={{ color: "#64748b", fontSize: "0.75rem", marginBottom: "0.5rem" }}>{g.locus} · {GENE_INFO[g.gene]?.size || "—"} · {GENE_INFO[g.gene]?.inh || "—"}</div>
                <div style={{ fontSize: "0.78rem", lineHeight: 1.6, marginBottom: "0.75rem", color: "#cbd5e1" }}>{g.protein_size}</div>

                <div style={{ marginBottom: "0.75rem" }}>
                  <div style={{ color: "#60a5fa", fontSize: "0.72rem", marginBottom: "0.4rem", fontWeight: 700 }}>KEY MUTATIONS</div>
                  {(g.key_mutations || []).map((m, i) => (
                    <div key={i} style={{ background: "#0f172a", borderRadius: "4px", padding: "0.5rem", marginBottom: "0.3rem", fontSize: "0.76rem" }}>
                      <span style={{ color: "#f59e0b" }}>{m.variant}</span> — <span style={{ color: "#94a3b8" }}>{m.protein_effect}</span> — <span style={{ color: "#64748b" }}>{m.location}</span><br />
                      <span style={{ color: "#cbd5e1" }}>{m.phenotype}</span>
                    </div>
                  ))}
                </div>

                <div>
                  <div style={{ color: "#4ade80", fontSize: "0.72rem", marginBottom: "0.4rem", fontWeight: 700 }}>SURVEILLANCE PROTOCOL</div>
                  {(g.surveillance || []).map((s, i) => (
                    <div key={i} style={{ background: "#0f172a", borderRadius: "4px", padding: "0.4rem 0.6rem", marginBottom: "0.25rem", fontSize: "0.76rem", color: "#e2e8f0" }}>&#x2022; {s}</div>
                  ))}
                </div>
              </div>
            ))}

            {/* Key concepts */}
            <h2 style={{ color: "#93c5fd", marginTop: "1.5rem", marginBottom: "0.75rem", fontSize: "1rem" }}>Key Clinical Concepts</h2>
            {Object.entries(keyConcepts).map(([k, v]) => (
              <div key={k} style={{ background: "#1e293b", border: "1px solid #334155", borderRadius: "6px", padding: "0.75rem", marginBottom: "0.5rem" }}>
                <div style={{ color: "#f59e0b", fontSize: "0.78rem", fontWeight: 700, marginBottom: "0.3rem" }}>{k}</div>
                <div style={{ fontSize: "0.78rem", lineHeight: 1.6, color: "#cbd5e1" }}>{v}</div>
              </div>
            ))}
          </div>
        )}

      </div>
    </div>
  );
}
