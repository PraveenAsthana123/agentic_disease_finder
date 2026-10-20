'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-jak3-il7r-cd25-zap70-cd45-il2rb-dock2-sp110-cid-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'JAK3':  '#b71c1c',  // deep red      — JAK3-SCID, T-B+NK-, γc downstream, autosomal X-SCID phenocopy
  'IL7RA': '#e65100',  // deep orange   — IL7RA-SCID, T-B+NK+, T only absent NK intact
  'IL2RA': '#1565c0',  // deep blue     — CD25/IL2RA, Treg deficiency, paradox lymphoproliferation
  'ZAP70': '#4a148c',  // deep purple   — ZAP70 selective CD8 absence PATHOGNOMONIC
  'PTPRC': '#00695c',  // dark teal     — CD45/PTPRC T-B-NK- SCID, CD45 absent ALL lymphocytes
  'IL2RB': '#f9a825',  // amber         — IL2RB/CD122, NK absent + T lymphoproliferation, HLH-like
  'DOCK2': '#2e7d32',  // dark green    — DOCK2, NK degranulation absent, viral susceptibility
  'SP110': '#6a1b9a',  // purple        — SP110/VODI, spontaneous hepatic VOD + immunodeficiency
};

const GENE_INFO = {
  'JAK3':  { full: 'JAK3 / Janus Kinase 3 / 1124aa', locus: '19p13.11', size: '1124 aa / 125 kDa (constitutively bound to γc chain; transduces IL-2/IL-4/IL-7/IL-9/IL-15/IL-21 signals; T-B+NK- SCID both sexes; autosomal X-SCID phenocopy)', inh: 'AR' },
  'IL7RA': { full: 'IL7RA / IL-7 Receptor Subunit Alpha (CD127) / 459aa', locus: '5p13.2', size: '459 aa / 52 kDa (IL-7Rα pairs with γc for IL-7 only; T cell thymic development and naive survival; T-B+NK+ SCID; NK intact via IL-15Rβγ)', inh: 'AR' },
  'IL2RA': { full: 'IL2RA / IL-2 Receptor Subunit Alpha (CD25) / 272aa', locus: '10p15.1', size: '272 aa / 30 kDa (IL-2Rα high-affinity subunit; Treg IL-2 signaling essential; CD25 LOF = Treg absent → lymphoproliferation + infections; paradox immunodeficiency)', inh: 'AR' },
  'ZAP70': { full: 'ZAP70 / Zeta-Chain Associated Protein Kinase 70 / 619aa', locus: '2q11.2', size: '619 aa / 70 kDa (TCR proximal kinase; CD8-LCK→ZAP70 higher threshold than CD4-LCK; selective CD8 absence PATHOGNOMONIC; autoimmunity in CD4 cells)', inh: 'AR' },
  'PTPRC': { full: 'PTPRC / Protein Tyrosine Phosphatase Receptor Type C (CD45) / 1304aa', locus: '1q31.3', size: '1304 aa / 147 kDa (largest leukocyte surface glycoprotein; dephosphorylates LCK-Tyr505; all lymphocytes require CD45; T-B-NK- SCID with neutrophils present)', inh: 'AR' },
  'IL2RB': { full: 'IL2RB / IL-2/IL-15 Receptor Subunit Beta (CD122) / 525aa', locus: '22q12.3', size: '525 aa / 58 kDa (shared β-chain IL-2R + IL-15R; NK survival via IL-15Rβγ; CD122 LOF = NK absent + T lymphoproliferation + HLH-like; 2019 discovery)', inh: 'AR' },
  'DOCK2': { full: 'DOCK2 / Dedicator of Cytokinesis 2 / 1862aa', locus: '5q35.1', size: '1862 aa / 211 kDa (atypical RAC1/RAC2 GEF; lymphocyte migration + NK degranulation + immune synapse actin; CD107a assay negative despite NK present; NEJM 2015)', inh: 'AR' },
  'SP110': { full: 'SP110 / SP110 Nuclear Body Protein / 697aa', locus: '2q37.1', size: '697 aa / 79 kDa (PML nuclear body scaffold; IFN-γ-stimulated gene regulation; hepatic venule endothelium + lymphoid; VODI = spontaneous infant VOD + CID; Druze founder)', inh: 'AR' },
};

const PATHOGNOMONIC = {
  'JAK3':  'T-B+NK- SCID in BOTH SEXES — autosomal equivalent of IL2RG X-SCID; pSTAT5 absent after all γc cytokines (IL-2/IL-7/IL-15); BCG ABSOLUTELY CI',
  'IL7RA': 'T-B+NK+ SCID — NK cells PRESENT (T absent only); distinguishes from JAK3/IL2RG (T-B+NK-); pSTAT5 absent after IL-7 ONLY, intact after IL-2/IL-15',
  'IL2RA': 'CD25 ABSENT on activated T cells (flow) + lymphoproliferation + CMV + IBD — paradox immunodeficiency; Tregs (CD25+FOXP3+) absent/non-functional',
  'ZAP70': 'CD8+ T cells ABSENT (< 50/μL) + CD4 NORMAL/ELEVATED — THE ONLY PID WITH THIS PATTERN; pZAP70 absent after CD3 cross-linking; autoimmunity common',
  'PTPRC': 'CD45 ABSENT on ALL lymphocytes by flow (anti-CD45-FITC 0% positive) — T-B-NK- SCID with neutrophils PRESENT; simplest pathognomonic flow test in PID',
  'IL2RB': 'NK CELLS ABSENT + T cell expansion + HLH-like (hyperferritinemia + hemophagocytosis) — NK absent distinguishes from FHL; pSTAT5 absent after IL-2 and IL-15',
  'DOCK2': 'NK degranulation ABSENT (CD107a assay negative) + NK cells NUMERICALLY NORMAL — severe CMV+HSV+RSV triad; NO giant granules (distinguishes from LYST/CHS)',
  'SP110': 'Spontaneous HEPATIC VOD in infant WITHOUT prior HSCT + hypogammaglobulinemia + lymphopenia — PATHOGNOMONIC for VODI; Doppler: reversed hepatic venous flow',
};

function GeneChip({ gene, active, onClick }) {
  return (
    <button
      onClick={() => onClick(gene)}
      style={{
        background: active ? GENE_COLORS[gene] : '#1e293b',
        color: '#fff',
        border: `2px solid ${GENE_COLORS[gene]}`,
        borderRadius: 8,
        padding: '6px 14px',
        margin: 4,
        cursor: 'pointer',
        fontWeight: active ? 700 : 400,
        fontSize: 13,
      }}
    >
      {gene}
    </button>
  );
}

export default function HereditaryJAK3IL7RCIDAtlas() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeGene, setActiveGene] = useState('JAK3');

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  const genes = ['JAK3', 'IL7RA', 'IL2RA', 'ZAP70', 'PTPRC', 'IL2RB', 'DOCK2', 'SP110'];

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#e2e8f0', fontFamily: 'monospace', padding: '24px' }}>
      <div style={{ maxWidth: 1200, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ borderBottom: '2px solid #1e3a5f', paddingBottom: 16, marginBottom: 20 }}>
          <h1 style={{ color: '#38bdf8', fontSize: 22, margin: 0 }}>
            🧬 Hereditary-JAK3-IL7R-CD25-ZAP70-CD45-IL2RB-DOCK2-SP110-CID-Variant-Atlas
          </h1>
          <p style={{ color: '#94a3b8', fontSize: 13, margin: '6px 0 0' }}>
            Complete 8-Gene Reference — Cytokine-Receptor-Signaling &amp; Proximal TCR/BCR Defects Causing Distinct CID/SCID-Variant Phenotypes · IUIS 2022 · 320 patients (8×40) · seeds 4054–4061
          </p>
        </div>

        {/* Gene Chips */}
        <div style={{ marginBottom: 16 }}>
          {genes.map(g => (
            <GeneChip key={g} gene={g} active={activeGene === g} onClick={setActiveGene} />
          ))}
        </div>

        {/* Active Gene Info */}
        {activeGene && (
          <div style={{ background: '#1e293b', border: `2px solid ${GENE_COLORS[activeGene]}`, borderRadius: 10, padding: '14px 18px', marginBottom: 18 }}>
            <div style={{ color: GENE_COLORS[activeGene], fontWeight: 700, fontSize: 15 }}>{GENE_INFO[activeGene].full}</div>
            <div style={{ color: '#94a3b8', fontSize: 12, marginTop: 4 }}>
              <span style={{ color: '#38bdf8' }}>Locus:</span> {GENE_INFO[activeGene].locus} &nbsp;|&nbsp;
              <span style={{ color: '#38bdf8' }}>Size:</span> {GENE_INFO[activeGene].size} &nbsp;|&nbsp;
              <span style={{ color: '#38bdf8' }}>Inheritance:</span> {GENE_INFO[activeGene].inh}
            </div>
            <div style={{ color: '#fbbf24', fontSize: 12, marginTop: 6 }}>
              <strong>PATHOGNOMONIC:</strong> {PATHOGNOMONIC[activeGene]}
            </div>
          </div>
        )}

        {/* Tabs */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
          {TABS.map((t, i) => (
            <button
              key={i}
              onClick={() => setTab(i)}
              style={{
                background: tab === i ? '#1e3a5f' : '#1e293b',
                color: tab === i ? '#38bdf8' : '#94a3b8',
                border: tab === i ? '1px solid #38bdf8' : '1px solid #334155',
                borderRadius: 6,
                padding: '6px 16px',
                cursor: 'pointer',
                fontSize: 13,
                fontWeight: tab === i ? 700 : 400,
              }}
            >{t}</button>
          ))}
        </div>

        {loading && <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>Loading atlas data…</div>}
        {error && <div style={{ color: '#f87171', padding: 20 }}>Error: {error}</div>}

        {/* Tab 0: Overview */}
        {!loading && !error && tab === 0 && overview && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 14, marginBottom: 20 }}>
              {genes.map(g => {
                const c = overview.cohorts?.[g];
                if (!c) return null;
                return (
                  <div key={g} style={{ background: '#1e293b', border: `1px solid ${GENE_COLORS[g]}`, borderRadius: 8, padding: 14 }}>
                    <div style={{ color: GENE_COLORS[g], fontWeight: 700, fontSize: 14, marginBottom: 6 }}>{g}</div>
                    <div style={{ fontSize: 12, color: '#94a3b8' }}>n={c.n} · seed {c.seed}</div>
                    <div style={{ fontSize: 12, marginTop: 4 }}>
                      <span style={{ color: '#f87171' }}>T-lymphopenia:</span> {c.t_lymphopenia_pct?.toFixed(1)}%
                    </div>
                    <div style={{ fontSize: 12 }}>
                      <span style={{ color: '#fb923c' }}>NK absent:</span> {c.nk_absent_pct?.toFixed(1)}%
                    </div>
                    <div style={{ fontSize: 12 }}>
                      <span style={{ color: '#34d399' }}>B cells present:</span> {c.b_present_pct?.toFixed(1)}%
                    </div>
                    <div style={{ fontSize: 12 }}>
                      <span style={{ color: '#818cf8' }}>Autoimmunity:</span> {c.autoimmune_pct?.toFixed(1)}%
                    </div>
                    <div style={{ fontSize: 12 }}>
                      <span style={{ color: '#38bdf8' }}>Viral suscept.:</span> {c.viral_suscept_pct?.toFixed(1)}%
                    </div>
                    <div style={{ fontSize: 12 }}>
                      <span style={{ color: '#fbbf24' }}>HSCT rate:</span> {c.hsct_pct?.toFixed(1)}%
                    </div>
                    <div style={{ fontSize: 12 }}>
                      <span style={{ color: '#a78bfa' }}>Survival 5yr:</span> {c.survival_5yr_pct?.toFixed(1)}%
                    </div>
                  </div>
                );
              })}
            </div>

            <div style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 14 }}>
              <div style={{ color: '#38bdf8', fontWeight: 700, marginBottom: 8 }}>Pathway Overview</div>
              <div style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.7 }}>{overview.pathway_overview}</div>
            </div>

            <div style={{ background: '#1e293b', borderRadius: 8, padding: 16 }}>
              <div style={{ color: '#38bdf8', fontWeight: 700, marginBottom: 8 }}>Key Clinical Facts</div>
              {Object.entries(overview.key_facts || {}).map(([k, v]) => (
                <div key={k} style={{ marginBottom: 8 }}>
                  <span style={{ color: '#fbbf24', fontWeight: 600, textTransform: 'uppercase', fontSize: 11 }}>{k.replace(/_/g, ' ')}: </span>
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tab 1: Gene Table */}
        {!loading && !error && tab === 1 && breakdown && (
          <div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#1e3a5f', color: '#38bdf8' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Gene</th>
                    <th style={{ padding: '8px 12px' }}>T-Lymphopenia</th>
                    <th style={{ padding: '8px 12px' }}>NK Absent</th>
                    <th style={{ padding: '8px 12px' }}>B Present</th>
                    <th style={{ padding: '8px 12px' }}>Autoimmune</th>
                    <th style={{ padding: '8px 12px' }}>Viral Suscept.</th>
                    <th style={{ padding: '8px 12px' }}>HSCT Rate</th>
                    <th style={{ padding: '8px 12px' }}>Survival 5yr</th>
                  </tr>
                </thead>
                <tbody>
                  {genes.map((g, i) => {
                    const c = breakdown.gene_summary?.[g];
                    if (!c) return null;
                    return (
                      <tr key={g} style={{ background: i % 2 === 0 ? '#1e293b' : '#0f172a', cursor: 'pointer' }}
                        onClick={() => setActiveGene(g)}>
                        <td style={{ padding: '8px 12px', color: GENE_COLORS[g], fontWeight: 700 }}>{g}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'center', color: '#f87171' }}>{c.t_lymphopenia_pct?.toFixed(1)}%</td>
                        <td style={{ padding: '8px 12px', textAlign: 'center', color: '#fb923c' }}>{c.nk_absent_pct?.toFixed(1)}%</td>
                        <td style={{ padding: '8px 12px', textAlign: 'center', color: '#34d399' }}>{c.b_present_pct?.toFixed(1)}%</td>
                        <td style={{ padding: '8px 12px', textAlign: 'center', color: '#818cf8' }}>{c.autoimmune_pct?.toFixed(1)}%</td>
                        <td style={{ padding: '8px 12px', textAlign: 'center', color: '#38bdf8' }}>{c.viral_suscept_pct?.toFixed(1)}%</td>
                        <td style={{ padding: '8px 12px', textAlign: 'center', color: '#fbbf24' }}>{c.hsct_pct?.toFixed(1)}%</td>
                        <td style={{ padding: '8px 12px', textAlign: 'center', color: '#a78bfa' }}>{c.survival_5yr_pct?.toFixed(1)}%</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Diagnostic Groups */}
            <div style={{ marginTop: 20 }}>
              <div style={{ color: '#38bdf8', fontWeight: 700, fontSize: 15, marginBottom: 12 }}>Diagnostic Groups by Mechanism</div>
              {Object.entries(breakdown.diagnostic_groups || {}).map(([gk, gv]) => (
                <div key={gk} style={{ background: '#1e293b', borderRadius: 8, padding: 14, marginBottom: 12 }}>
                  <div style={{ color: '#fbbf24', fontWeight: 700, fontSize: 13, marginBottom: 6 }}>
                    {gk.replace(/_/g, ' ').toUpperCase()} — Genes: [{(gv.genes || []).join(', ')}]
                  </div>
                  <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 6 }}><strong style={{ color: '#38bdf8' }}>Mechanism:</strong> {gv.mechanism}</div>
                  <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 6 }}><strong style={{ color: '#34d399' }}>Shared Feature:</strong> {gv.shared_feature}</div>
                  <div style={{ color: '#94a3b8', fontSize: 12 }}><strong style={{ color: '#f87171' }}>Key DDx:</strong> {gv.key_ddx}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tab 2: Clinical Atlas */}
        {!loading && !error && tab === 2 && breakdown && (
          <div>
            <div style={{ color: '#38bdf8', fontWeight: 700, fontSize: 15, marginBottom: 14 }}>
              CID-Variant Pattern Map — Flow Cytometry Triage
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 14 }}>
              {genes.map(g => (
                <div key={g} style={{ background: '#1e293b', border: `1px solid ${GENE_COLORS[g]}40`, borderRadius: 8, padding: 14, cursor: 'pointer' }}
                  onClick={() => setActiveGene(g)}>
                  <div style={{ color: GENE_COLORS[g], fontWeight: 700, fontSize: 13, marginBottom: 6 }}>{g} — {GENE_INFO[g].full.split('/')[0].trim()} ({GENE_INFO[g].locus})</div>
                  <div style={{ color: '#94a3b8', fontSize: 11, lineHeight: 1.5 }}><strong style={{ color: '#fbbf24' }}>PATHOGNOMONIC:</strong> {PATHOGNOMONIC[g]}</div>
                  <div style={{ color: '#64748b', fontSize: 11, marginTop: 4 }}>{GENE_INFO[g].size}</div>
                </div>
              ))}
            </div>

            {/* Flow Cytometry Decision Matrix */}
            <div style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginTop: 20 }}>
              <div style={{ color: '#38bdf8', fontWeight: 700, marginBottom: 10 }}>Flow Cytometry Triage — T/B/NK Phenotype → Gene Priority</div>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#0f172a', color: '#38bdf8' }}>
                    <th style={{ padding: '6px 10px', textAlign: 'left' }}>T cells</th>
                    <th style={{ padding: '6px 10px', textAlign: 'left' }}>B cells</th>
                    <th style={{ padding: '6px 10px', textAlign: 'left' }}>NK cells</th>
                    <th style={{ padding: '6px 10px', textAlign: 'left' }}>CD45 on lymph.</th>
                    <th style={{ padding: '6px 10px', textAlign: 'left' }}>Top Gene(s)</th>
                    <th style={{ padding: '6px 10px', textAlign: 'left' }}>Key Distinguisher</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { t: 'Absent', b: 'Present', nk: 'Absent', cd45: 'Present', genes: 'JAK3 (or IL2RG)', key: 'pSTAT5 absent after IL-2/IL-7/IL-15; BCG absolutely CI' },
                    { t: 'Absent', b: 'Present', nk: 'Present', cd45: 'Present', genes: 'IL7RA', key: 'pSTAT5 absent after IL-7 ONLY; NK intact via IL-15Rβγ' },
                    { t: 'Absent', b: 'Absent', nk: 'Absent', cd45: 'ABSENT', genes: 'PTPRC / CD45', key: 'CD45 absent = PATHOGNOMONIC; neutrophils PRESENT (not RD)' },
                    { t: 'CD8 absent / CD4 normal', b: 'Present', nk: 'Present', cd45: 'Present', genes: 'ZAP70', key: 'Only PID with isolated CD8 absence; pZAP70 absent; autoimmunity' },
                    { t: 'Expanded + autoimmune', b: 'Present', nk: 'Present', cd45: 'Present', genes: 'IL2RA / CD25', key: 'CD25 absent on activated T cells; Tregs reduced; CMV + IBD' },
                    { t: 'Expanded + HLH', b: 'Present', nk: 'Absent', cd45: 'Present', genes: 'IL2RB / CD122', key: 'NK absent + T expansion + HLH-like; CD122 absent on T/NK flow' },
                    { t: 'Reduced function', b: 'Variable', nk: 'Present (low function)', cd45: 'Present', genes: 'DOCK2', key: 'CD107a degranulation ABSENT despite NK present; CMV+HSV+RSV triad' },
                    { t: 'Reduced', b: 'Reduced', nk: 'Reduced', cd45: 'Present', genes: 'SP110 / VODI', key: 'Spontaneous infant hepatic VOD + low IgG + lymphopenia' },
                  ].map((row, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#1e293b' : '#0f172a' }}>
                      <td style={{ padding: '6px 10px', color: '#f87171' }}>{row.t}</td>
                      <td style={{ padding: '6px 10px', color: '#34d399' }}>{row.b}</td>
                      <td style={{ padding: '6px 10px', color: '#fb923c' }}>{row.nk}</td>
                      <td style={{ padding: '6px 10px', color: '#94a3b8' }}>{row.cd45}</td>
                      <td style={{ padding: '6px 10px', color: '#fbbf24', fontWeight: 700 }}>{row.genes}</td>
                      <td style={{ padding: '6px 10px', color: '#94a3b8', fontSize: 11 }}>{row.key}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Treatment Matrix */}
            {breakdown.diagnostic_groups && (
              <div style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginTop: 14 }}>
                <div style={{ color: '#38bdf8', fontWeight: 700, marginBottom: 8 }}>Treatment & BCG Protocol</div>
                {Object.entries(breakdown.diagnostic_groups || {}).slice(0, 4).map(([k, v]) => (
                  <div key={k} style={{ marginBottom: 8, paddingBottom: 8, borderBottom: '1px solid #334155' }}>
                    <span style={{ color: '#fbbf24', fontWeight: 600, fontSize: 11 }}>[{(v.genes || []).join('+')}] </span>
                    <span style={{ color: '#94a3b8', fontSize: 11 }}>{v.key_ddx?.slice(0, 200)}…</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Tab 3: Definitions */}
        {!loading && !error && tab === 3 && definitions && (
          <div>
            {(definitions.definitions || []).map((def, i) => (
              <div key={i} style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 14 }}>
                <div style={{ color: '#38bdf8', fontWeight: 700, fontSize: 14, marginBottom: 8 }}>{def.term}</div>
                <div style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.75 }}>{def.definition}</div>
              </div>
            ))}

            {/* Key Distinguishers */}
            <div style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginTop: 14 }}>
              <div style={{ color: '#38bdf8', fontWeight: 700, marginBottom: 8 }}>Key Distinguishers — Quick Reference</div>
              {Object.entries(definitions.key_distinguishers || {}).map(([k, v]) => (
                <div key={k} style={{ marginBottom: 8, paddingBottom: 6, borderBottom: '1px solid #334155' }}>
                  <span style={{ color: '#fbbf24', fontWeight: 600, fontSize: 11, textTransform: 'uppercase' }}>{k.replace(/_/g, ' ')}: </span>
                  <span style={{ color: '#94a3b8', fontSize: 11 }}>{v}</span>
                </div>
              ))}
            </div>

            {/* Treatment Matrix */}
            {definitions.treatment_matrix && (
              <div style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginTop: 14 }}>
                <div style={{ color: '#38bdf8', fontWeight: 700, marginBottom: 8 }}>Treatment Matrix</div>
                {Object.entries(definitions.treatment_matrix).map(([k, v]) => (
                  <div key={k} style={{ marginBottom: 8 }}>
                    <span style={{ color: '#34d399', fontWeight: 600, fontSize: 11, textTransform: 'uppercase' }}>{k.replace(/_/g, ' ')}: </span>
                    <span style={{ color: '#94a3b8', fontSize: 11 }}>{v}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
