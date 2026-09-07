'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  DDC:     '#dc2626',  // red    — AADC; oculogyric crises; Upstaza FDA2023
  GCH1:    '#ea580c',  // orange — DRD/Segawa; diurnal variation; L-Dopa CURATIVE
  TH:      '#ca8a04',  // amber  — TH deficiency; BH4 NORMAL; infantile Parkinsonism
  SPR:     '#16a34a',  // green  — sepiapterin reductase; CSF sepiapterin PATHOGNOMONIC
  ALDH7A1: '#2563eb',  // blue   — PDE/antiquitin; alpha-AASA PATHOGNOMONIC; pyridoxine
  PNPO:    '#7c3aed',  // purple — PNPO; PLP MANDATORY NOT pyridoxine; fatal mistake
  SLC6A3:  '#0f766e',  // teal   — DTDS; DAT-SPECT absent; L-Dopa ABSOLUTELY CI
  GATM:    '#0e7490',  // cyan   — AGAT; GAA LOW; creatine 400mgkgday CURATIVE
};

const GENE_DISEASE = {
  DDC:     'AR AADC-Deficiency — DDC-480aa — 7p12.2 — Aromatic-L-Amino-Acid-Decarboxylase-PLP-Dependent — Oculogyric-Crises-PATHOGNOMONIC — HVA-LOW-5HIAA-LOW-Both — Upstaza-Eladocagene-FDA2023',
  GCH1:    'AD DRD-Segawa — GCH1-250aa — 14q22.2 — GTP-Cyclohydrolase-1-BH4-Rate-Limiting — Diurnal-Variation-PATHOGNOMONIC — L-Dopa-Ultra-Low-Dose-CURATIVE — Masquerades-as-CP',
  TH:      'AR TH-Deficiency — TH-498aa — 11p15.5 — Tyrosine-Hydroxylase-BH4-Dependent — BH4-NORMAL-KEY-DDx-from-GCH1-AR — Infantile-Parkinsonism-Dystonia — TH1-DRD-like-vs-TH2-Severe',
  SPR:     'AR Sepiapterin-Reductase-Deficiency — SPR-263aa — 2p14 — BH4-Final-Reduction-Step — CSF-Sepiapterin-PATHOGNOMONIC — L-Dopa-PLUS-5HTP-MANDATORY-COMBINATION — Urine-Biopterin-NOT-Elevated',
  ALDH7A1: 'AR Pyridoxine-Dependent-Epilepsy — ALDH7A1-539aa — 5q31.2 — Antiquitin-Lysine-Catabolism — alpha-AASA-Urine-PATHOGNOMONIC — Pyridoxine-NOT-PLP-First — Lysine-Restriction-Second',
  PNPO:    'AR PNPO-Deficiency — PNPO-261aa — 17q21.32 — Pyridoxamine-5-phosphate-Oxidase-PLP-Synthesis — PLP-MANDATORY-NOT-Pyridoxine — Burst-Suppression-EEG — Pyridoxine-FATAL-Mistake',
  SLC6A3:  'AR DTDS-DAT-Deficiency — SLC6A3-620aa — 5p15.33 — Dopamine-Transporter-Reuptake — DAT-SPECT-Absent-PATHOGNOMONIC — L-Dopa-ABSOLUTELY-CI-Worsens — HVA-HIGH-NOT-LOW',
  GATM:    'AR AGAT-Deficiency-Creatine-Step1 — GATM-423aa — 15q21.1 — Arginine-Glycine-Amidinotransferase — GAA-LOW-NOT-HIGH — MRS-Absent-Cr-Peak — Creatine-400mgkgday-CURATIVE',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#94a3b8' }}>Loading…</div>;
}

function ErrorBox({ msg }) {
  return (
    <div style={{ padding: '1rem', background: '#450a0a', borderRadius: 8, color: '#fca5a5', margin: '1rem 0' }}>
      Error: {msg}
    </div>
  );
}

function KPI({ label, value, color }) {
  return (
    <div style={{
      background: '#1e293b', borderRadius: 10, padding: '1rem 1.2rem',
      borderLeft: `4px solid ${color || '#6366f1'}`, minWidth: 160,
    }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: color || '#a5b4fc' }}>{value}</div>
      <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{label}</div>
    </div>
  );
}

function Alert({ text, level }) {
  const colors = { critical: '#fca5a5', warning: '#fcd34d', info: '#93c5fd' };
  const bg     = { critical: '#450a0a', warning: '#451a03', info: '#0c1a3a' };
  const lv = (text || '').includes('ABSOLUTELY-CI') || (text || '').includes('PATHOGNOMONIC') || (text || '').includes('MANDATORY') || (text || '').includes('FATAL') ? 'critical'
           : (text || '').includes('CI') || (text || '').includes('CURATIVE') || (text || '').includes('RESPONSIVE') ? 'warning' : 'info';
  return (
    <div style={{
      background: bg[lv], border: `1px solid ${colors[lv]}33`,
      borderLeft: `3px solid ${colors[lv]}`, borderRadius: 6,
      padding: '0.4rem 0.7rem', fontSize: 12, color: colors[lv], marginBottom: 4,
    }}>{text}</div>
  );
}

/* ── OVERVIEW TAB ─────────────────────────────────────────────────────────── */
function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const s = data.aggregate_stats || {};
  const gs = data.gene_summary || [];
  return (
    <div>
      <h2 style={{ color: '#f1f5f9', marginBottom: 4 }}>{data.atlas}</h2>
      <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: '1.5rem', lineHeight: 1.5 }}>
        {data.subtitle}
      </p>

      {/* KPIs */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: '1.5rem' }}>
        <KPI label="Total Patients"       value={s.total_patients}                                 color="#6366f1" />
        <KPI label="Genes Covered"        value={s.genes_covered || 8}                             color="#10b981" />
        <KPI label="Seeds"                value={s.seed_range}                                     color="#f59e0b" />
        <KPI label="Avg Age at Onset (yr)" value={s.avg_age_at_presentation_yr ?? '—'}             color="#3b82f6" />
        <KPI label="Severe Cases %"       value={`${s.severe_cases_pct ?? '—'}%`}                  color="#ef4444" />
      </div>

      {/* Gene summary cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 12, marginBottom: '1.5rem' }}>
        {gs.map(g => (
          <div key={g.gene} style={{
            background: '#1e293b', borderRadius: 10, padding: '0.85rem 1rem',
            borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
              <span style={{ fontWeight: 700, fontSize: 16, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</span>
              <span style={{ fontSize: 11, color: '#64748b', background: '#0f172a', padding: '2px 7px', borderRadius: 4 }}>{g.inheritance} · {g.locus}</span>
            </div>
            <div style={{ fontSize: 11, color: '#fcd34d', marginBottom: 4 }}>🧪 {g.key_biomarker}</div>
            <div style={{ fontSize: 11, color: '#f87171', marginBottom: 4 }}>⚡ {g.pathognomonic}</div>
            <div style={{ fontSize: 11, color: '#86efac' }}>💊 {g.treatment}</div>
            {(g.critical_flags || []).slice(0, 1).map((f, i) => (
              <div key={i} style={{ fontSize: 11, color: '#fca5a5', marginTop: 4 }}>⚠ {f}</div>
            ))}
          </div>
        ))}
      </div>

      {/* Key clinical distinctions */}
      {data.key_clinical_distinctions && data.key_clinical_distinctions.length > 0 && (
        <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1rem' }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#f1f5f9', marginBottom: 8 }}>Key DDx Distinctions</div>
          {data.key_clinical_distinctions.map((a, i) => <Alert key={i} text={a} />)}
        </div>
      )}
    </div>
  );
}

/* ── GENE TABLE TAB ───────────────────────────────────────────────────────── */
function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = data.genes || [];
  return (
    <div>
      <h3 style={{ color: '#f1f5f9', marginBottom: '1rem' }}>Per-Gene Breakdown — 40 Patients Each</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#0f172a' }}>
              {['Gene','Locus','Protein','Biomarker','Pathognomonic','Mild%','Mod%','Sev%','N'].map(h => (
                <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', borderBottom: '1px solid #1e293b' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {genes.map((g, i) => {
              const sv = g.severity_distribution || {};
              const total = g.n_patients || 40;
              const mp = v => `${Math.round(((v||0)/total)*100)}%`;
              return (
                <tr key={g.gene} style={{ background: i%2===0 ? '#0f172a' : '#1e293b' }}>
                  <td style={{ padding: '7px 10px', color: GENE_COLORS[g.gene] || '#a5b4fc', fontWeight: 700 }}>{g.gene}</td>
                  <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{g.locus}</td>
                  <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{g.protein_size}</td>
                  <td style={{ padding: '7px 10px', color: '#fcd34d', maxWidth: 200 }}>{g.key_biomarker}</td>
                  <td style={{ padding: '7px 10px', color: '#f87171', maxWidth: 220 }}>{g.pathognomonic}</td>
                  <td style={{ padding: '7px 10px', color: '#10b981' }}>{mp(sv.mild)}</td>
                  <td style={{ padding: '7px 10px', color: '#f59e0b' }}>{mp(sv.moderate)}</td>
                  <td style={{ padding: '7px 10px', color: '#ef4444' }}>{mp(sv.severe)}</td>
                  <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{total}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Critical flags per gene */}
      <div style={{ marginTop: '1.5rem' }}>
        <h4 style={{ color: '#f1f5f9', marginBottom: '0.75rem' }}>Critical Flags Per Gene</h4>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(360px,1fr))', gap: 12 }}>
          {genes.map(g => (
            <div key={g.gene} style={{ background: '#1e293b', borderRadius: 10, padding: '0.85rem 1rem', borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}` }}>
              <div style={{ fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc', marginBottom: 6 }}>{g.gene}</div>
              {(g.critical_flags || []).map((f, i) => <Alert key={i} text={f} />)}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ── CLINICAL ATLAS TAB ───────────────────────────────────────────────────── */
function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const genes = data.genes || [];
  return (
    <div>
      <h3 style={{ color: '#f1f5f9', marginBottom: '1rem' }}>Clinical Atlas — Neurotransmitter Synthesis Disorders</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(480px,1fr))', gap: 14 }}>
        {genes.map(g => (
          <div key={g.gene} style={{
            background: '#1e293b', borderRadius: 10, padding: '1rem',
            borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
              <span style={{ fontWeight: 700, fontSize: 16, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</span>
              <div style={{ display: 'flex', gap: 6 }}>
                <span style={{ fontSize: 10, color: '#64748b', background: '#0f172a', padding: '2px 6px', borderRadius: 4 }}>{g.locus}</span>
                <span style={{ fontSize: 10, color: '#64748b', background: '#0f172a', padding: '2px 6px', borderRadius: 4 }}>{g.protein_size}</span>
                <span style={{ fontSize: 10, color: '#64748b', background: '#0f172a', padding: '2px 6px', borderRadius: 4 }}>{g.inheritance}</span>
              </div>
            </div>
            <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.5, marginBottom: 8 }}>
              {(GENE_DISEASE[g.gene] || '').replace(/--/g, '·').replace(/_/g, ' ')}
            </div>
            <div style={{ fontSize: 11, color: '#fcd34d', marginBottom: 4 }}>🧪 <b>Biomarker:</b> {g.key_biomarker}</div>
            <div style={{ fontSize: 11, color: '#f87171', marginBottom: 4 }}>⚡ <b>Pathognomonic:</b> {g.pathognomonic}</div>
            <div style={{ fontSize: 11, color: '#86efac', marginBottom: 4 }}>💊 <b>Treatment:</b> {g.treatment}</div>
            <div style={{ marginTop: 8 }}>
              {(g.critical_flags || []).map((f, i) => <Alert key={i} text={f} />)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── DEFINITIONS TAB ──────────────────────────────────────────────────────── */
function DefinitionsTab({ data }) {
  const [open, setOpen] = useState(null);
  const [glossOpen, setGlossOpen] = useState(false);
  if (!data) return <Loading />;
  const genes = data.genes || [];
  const glossary = data.glossary || {};
  return (
    <div>
      <h3 style={{ color: '#f1f5f9', marginBottom: '1rem' }}>Gene Definitions — 8 Neurotransmitter Synthesis Genes</h3>
      {genes.map(d => (
        <div key={d.gene} style={{ background: '#1e293b', borderRadius: 10, marginBottom: 10, overflow: 'hidden' }}>
          <button
            onClick={() => setOpen(open === d.gene ? null : d.gene)}
            style={{
              width: '100%', textAlign: 'left', padding: '0.9rem 1rem', background: 'transparent',
              border: 'none', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            }}
          >
            <span style={{ fontWeight: 700, color: GENE_COLORS[d.gene] || '#a5b4fc', fontSize: 15 }}>
              {d.gene} — {d.protein_size} · {d.locus} · {d.inheritance}
            </span>
            <span style={{ color: '#64748b' }}>{open === d.gene ? '▲' : '▼'}</span>
          </button>
          {open === d.gene && (
            <div style={{ padding: '0 1rem 1rem', borderTop: '1px solid #0f172a' }}>
              <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.6, margin: '0.7rem 0' }}>
                {d.definition}
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', marginBottom: '0.75rem', fontSize: 12 }}>
                {[
                  ['Chromosome', d.locus],
                  ['Protein Size', d.protein_size],
                  ['Inheritance', d.inheritance],
                  ['Age of Onset', d.age_of_onset],
                ].map(([lbl, val]) => (
                  <div key={lbl} style={{ background: '#0f172a', borderRadius: 6, padding: '0.4rem 0.6rem' }}>
                    <div style={{ color: '#64748b', fontSize: 10 }}>{lbl}</div>
                    <div style={{ color: '#cbd5e1' }}>{val}</div>
                  </div>
                ))}
              </div>
              {(d.critical_flags || []).length > 0 && (
                <div>
                  <div style={{ fontSize: 12, fontWeight: 700, color: '#fca5a5', marginBottom: 4 }}>Critical Clinical Flags</div>
                  {(d.critical_flags || []).map((f, i) => <Alert key={i} text={f} />)}
                </div>
              )}
            </div>
          )}
        </div>
      ))}

      {/* Glossary */}
      <div style={{ background: '#1e293b', borderRadius: 10, marginTop: '1.5rem', overflow: 'hidden' }}>
        <button
          onClick={() => setGlossOpen(!glossOpen)}
          style={{
            width: '100%', textAlign: 'left', padding: '0.9rem 1rem', background: 'transparent',
            border: 'none', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          }}
        >
          <span style={{ fontWeight: 700, color: '#a5b4fc', fontSize: 15 }}>
            Neurotransmitter Glossary ({Object.keys(glossary).length} terms)
          </span>
          <span style={{ color: '#64748b' }}>{glossOpen ? '▲' : '▼'}</span>
        </button>
        {glossOpen && (
          <div style={{ padding: '0 1rem 1rem', borderTop: '1px solid #0f172a' }}>
            {Object.entries(glossary).map(([term, def]) => (
              <div key={term} style={{ marginTop: 8, fontSize: 12 }}>
                <span style={{ color: '#a5b4fc', fontWeight: 700 }}>{term}: </span>
                <span style={{ color: '#94a3b8' }}>{def}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* ── MAIN PAGE ────────────────────────────────────────────────────────────── */
export default function HereditaryNeurotransmitterAtlasPage() {
  const [tab, setTab]       = useState('Overview');
  const [overview, setOv]   = useState(null);
  const [breakdown, setBk]  = useState(null);
  const [defs, setDefs]     = useState(null);
  const [error, setError]   = useState(null);

  useEffect(() => {
    const base = `${API}/api/hereditary-neurotransmitter-atlas`;
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => { setOv(ov); setBk(bk); setDefs(df); })
      .catch(e => setError(e.message));
  }, []);

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#f1f5f9', fontFamily: 'Inter,sans-serif' }}>
      <div style={{ maxWidth: 1280, margin: '0 auto', padding: '1.5rem' }}>

        {/* Header */}
        <div style={{ marginBottom: '1.5rem' }}>
          <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>
            Hereditary Metabolic Disease Atlas Series · Inborn Errors of Neurotransmitter Metabolism
          </div>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: '#f1f5f9', margin: 0, lineHeight: 1.3 }}>
            🧬 Hereditary-Neurotransmitter-Atlas
          </h1>
          <div style={{ fontSize: 13, color: '#94a3b8', marginTop: 4 }}>
            Complete 8-Gene Hereditary Neurotransmitter Synthesis Disorders Atlas · 320 Patients (8×40, Seeds 1878–1885)
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
            {Object.entries(GENE_COLORS).map(([g, c]) => (
              <span key={g} style={{ fontSize: 11, padding: '2px 8px', borderRadius: 12, background: c + '22', color: c, border: `1px solid ${c}44` }}>
                {g}
              </span>
            ))}
          </div>
        </div>

        {error && <ErrorBox msg={error} />}

        {/* Tabs */}
        <div style={{ display: 'flex', gap: 4, marginBottom: '1.5rem', borderBottom: '1px solid #1e293b', paddingBottom: '0.5rem' }}>
          {TABS.map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              style={{
                padding: '0.5rem 1rem', background: tab === t ? '#6366f1' : 'transparent',
                color: tab === t ? '#fff' : '#94a3b8', border: 'none', borderRadius: 6,
                cursor: 'pointer', fontSize: 13, fontWeight: tab === t ? 600 : 400,
              }}
            >{t}</button>
          ))}
        </div>

        {/* Tab content */}
        {tab === 'Overview'      && <OverviewTab      data={overview}  />}
        {tab === 'Gene Table'    && <GeneTableTab     data={breakdown} />}
        {tab === 'Clinical Atlas'&& <ClinicalAtlasTab data={breakdown} />}
        {tab === 'Definitions'   && <DefinitionsTab   data={defs}      />}
      </div>
    </div>
  );
}
