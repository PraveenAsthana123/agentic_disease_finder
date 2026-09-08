'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  MC4R:  '#1565c0',  // deep blue    — most common monogenic obesity
  LEPR:  '#0d6e3d',  // deep green   — leptin receptor / setmelanotide
  LEP:   '#b71c1c',  // deep red     — leptin deficiency / metreleptin CURATIVE
  PCSK1: '#880e4f',  // deep magenta — panhypopituitarism / neonatal diarrhoea
  POMC:  '#e65100',  // deep orange  — red hair triad / adrenal crisis
  SH2B1: '#4a148c',  // deep purple  — 16p11.2 / behavioural
  KSR2:  '#006064',  // deep teal    — low HR / metformin UNIQUE
  SIM1:  '#1b5e20',  // forest green — PWS-like / OXT axis
};

const GENE_INFO = {
  MC4R:  { aa: 332,  locus: '18q21.32', inh: 'AD',       disease: 'Most-Common-Monogenic-Obesity-1-2pct — Hyperinsulinaemia-Early-Marker — Tall-Stature-Childhood — Setmelanotide-Partial-Benefit' },
  LEPR:  { aa: 1165, locus: '1p31.3',   inh: 'AR',       disease: 'Severe-Infantile-Hyperphagia — Very-High-Leptin-Paradox — Setmelanotide-FDA-2020 — Hypogonadism-Immune-Dysfunction' },
  LEP:   { aa: 167,  locus: '7q32.1',   inh: 'AR',       disease: 'Undetectable-Leptin-Pathognomonic — Metreleptin-CURATIVE — NOT-Setmelanotide — Normal-Cortisol-Normal-Hair' },
  PCSK1: { aa: 753,  locus: '5q15',     inh: 'AR',       disease: 'Neonatal-Diarrhoea-First — Panhypopituitarism — Setmelanotide-FDA-2021 — Adrenal-Crisis-Risk-Stress-Dosing' },
  POMC:  { aa: 267,  locus: '2p23.3',   inh: 'AR',       disease: 'Red-Hair-Obesity-Hypocortisolism-TRIAD — Setmelanotide-FDA-2020 — Adrenal-Crisis-MANDATORY-Cortisol — Sunscreen-MC1R-Absent' },
  SH2B1: { aa: 613,  locus: '16p11.2',  inh: 'AD',       disease: '16p11.2-Deletion-Microarray-Mandatory — Behavioural-Dysregulation-Aggression — Severe-Insulin-Resistance — NOT-Setmelanotide' },
  KSR2:  { aa: 950,  locus: '12q24.22', inh: 'AR/cpHet', disease: 'Low-Resting-HR-Distinctive — Metformin-DRAMATICALLY-Effective-UNIQUE — Severe-Insulin-Resistance — NOT-Setmelanotide' },
  SIM1:  { aa: 786,  locus: '6q16.3',   inh: 'AD',       disease: 'PWS-Like-Normal-15q-CRITICAL-DDx — Aggressive-Food-Seeking — OXT-Trials-PVN-Axis — 6q16.3-Microarray' },
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
function Alert({ text }) {
  const lv = /ABSOLUTELY.CI|PATHOGNOMONIC|MANDATORY|CONTRAINDICATED|CURATIVE|CRISIS|LETHAL|UNIQUE/i.test(text) ? 'critical'
           : /\bCI\b|RISK|MONITOR|WARNING|AVOID|CAUTION|NOT-INDICATED/i.test(text) ? 'warning' : 'info';
  const colors = { critical: '#fca5a5', warning: '#fcd34d', info: '#93c5fd' };
  const bg     = { critical: '#450a0a', warning: '#451a03', info: '#0c1a3a' };
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
  const geneCounts = data.gene_patient_counts || {};
  return (
    <div>
      <h2 style={{ color: '#f1f5f9', marginBottom: 4 }}>{data.atlas}</h2>
      <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: '1.5rem', lineHeight: 1.5 }}>
        {data.subtitle}
      </p>

      {/* KPIs */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: '1.5rem' }}>
        <KPI label="Total Patients"              value={data.total_patients}                    color="#6366f1" />
        <KPI label="Genes Covered"               value={(data.genes || []).length || 8}         color="#10b981" />
        <KPI label="Seeds"                       value={data.seeds}                             color="#f59e0b" />
        <KPI label="Hyperphagia Pts"             value={data.hyperphagia_patients}              color="#ef4444" />
        <KPI label="Severe Obesity Pts"          value={data.severe_obesity_patients}           color="#dc2626" />
        <KPI label="Hyperinsulinaemia Pts"       value={data.hyperinsulinaemia_patients}        color="#f97316" />
        <KPI label="Adrenal Insufficiency Pts"   value={data.adrenal_insufficiency_patients}    color="#b91c1c" />
        <KPI label="Red Hair Pts (POMC)"         value={data.red_hair_patients}                 color="#d97706" />
        <KPI label="Hypogonadism Pts"            value={data.hypogonadism_patients}             color="#7c3aed" />
        <KPI label="Setmelanotide Eligible"      value={data.setmelanotide_eligible_patients}   color="#0ea5e9" />
        <KPI label="T2D Pts"                     value={data.t2d_patients}                      color="#be123c" />
        <KPI label="Behavioural Issues Pts"      value={data.behavioural_issues_patients}       color="#4a148c" />
        <KPI label="Immune Dysfunction Pts"      value={data.immune_dysfunction_patients}       color="#0d6e3d" />
        <KPI label="Low HR Pts (KSR2)"           value={data.low_heart_rate_patients}           color="#006064" />
      </div>

      {/* Inheritance bar */}
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1.5rem' }}>
        <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>Inheritance Pattern (8 genes)</div>
        <div style={{ display: 'flex', height: 20, borderRadius: 6, overflow: 'hidden', gap: 2 }}>
          {[
            { label: 'AD (MC4R, SH2B1, SIM1)', val: 3, color: '#3b82f6' },
            { label: 'AR (LEPR, LEP, PCSK1, POMC)', val: 4, color: '#10b981' },
            { label: 'AR/cpHet (KSR2)', val: 1, color: '#f59e0b' },
          ].map(b => (
            <div key={b.label} title={b.label} style={{
              flex: b.val, background: b.color, display: 'flex', alignItems: 'center',
              justifyContent: 'center', fontSize: 10, color: '#fff', fontWeight: 600,
            }}>{b.val}</div>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 12, marginTop: 6, flexWrap: 'wrap' }}>
          {[
            { label: 'AD (MC4R, SH2B1, SIM1)', color: '#3b82f6' },
            { label: 'AR (LEPR, LEP, PCSK1, POMC)', color: '#10b981' },
            { label: 'AR/cpHet (KSR2)', color: '#f59e0b' },
          ].map(b => (
            <div key={b.label} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, color: '#94a3b8' }}>
              <div style={{ width: 8, height: 8, borderRadius: 2, background: b.color }} />{b.label}
            </div>
          ))}
        </div>
      </div>

      {/* Pathway */}
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1.5rem' }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: '#a5b4fc', marginBottom: 6 }}>Leptin-Melanocortin Pathway</div>
        <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.7 }}>{data.pathway}</div>
      </div>

      {/* Key insight */}
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1.5rem' }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: '#a5b4fc', marginBottom: 6 }}>Key Clinical Insights</div>
        <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.8 }}>{data.key_clinical_insight}</div>
      </div>

      {/* Per-gene bar */}
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem' }}>
        <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 10 }}>Patients per Gene (40 each)</div>
        {Object.entries(geneCounts).map(([gn, cnt]) => (
          <div key={gn} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
            <div style={{ width: 60, fontSize: 11, fontWeight: 700, color: GENE_COLORS[gn] || '#a5b4fc' }}>{gn}</div>
            <div style={{ flex: 1, background: '#0f172a', borderRadius: 4, height: 14, overflow: 'hidden' }}>
              <div style={{ width: `${(cnt / 40) * 100}%`, background: GENE_COLORS[gn] || '#6366f1', height: '100%', borderRadius: 4 }} />
            </div>
            <div style={{ width: 30, fontSize: 11, color: '#64748b', textAlign: 'right' }}>{cnt}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── GENE TABLE TAB ───────────────────────────────────────────────────────── */
function GeneTableTab() {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
        <thead>
          <tr style={{ background: '#1e293b' }}>
            {['Gene', 'aa', 'Locus', 'Inh.', 'Clinical Hallmark'].map(h => (
              <th key={h} style={{ padding: '8px 10px', color: '#94a3b8', textAlign: 'left', borderBottom: '1px solid #334155' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Object.entries(GENE_INFO).map(([gn, info]) => (
            <tr key={gn} style={{ borderBottom: '1px solid #1e293b' }}>
              <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[gn] || '#a5b4fc' }}>{gn}</td>
              <td style={{ padding: '8px 10px', color: '#cbd5e1' }}>{info.aa}</td>
              <td style={{ padding: '8px 10px', color: '#64748b', fontFamily: 'monospace' }}>{info.locus}</td>
              <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{info.inh}</td>
              <td style={{ padding: '8px 10px', color: '#94a3b8', lineHeight: 1.5 }}>{info.disease}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ── CLINICAL ATLAS TAB ───────────────────────────────────────────────────── */
function ClinicalAtlasTab({ data }) {
  const genes = data ? Object.keys(data) : Object.keys(GENE_INFO);
  const [selGene, setSelGene] = useState(genes[0] || 'MC4R');
  if (!data) return <Loading />;
  const g = data[selGene] || {};
  return (
    <div>
      {/* Gene selector */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: '1rem' }}>
        {genes.map(gn => (
          <button key={gn} onClick={() => setSelGene(gn)} style={{
            padding: '6px 14px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: 600,
            background: selGene === gn ? (GENE_COLORS[gn] || '#6366f1') : '#1e293b',
            color: selGene === gn ? '#fff' : '#94a3b8',
          }}>{gn}</button>
        ))}
      </div>

      {/* Gene panel */}
      <div style={{ background: '#1e293b', borderRadius: 12, padding: '1.2rem', borderLeft: `4px solid ${GENE_COLORS[selGene] || '#6366f1'}` }}>
        <div style={{ fontSize: 16, fontWeight: 700, color: GENE_COLORS[selGene] || '#a5b4fc', marginBottom: 4 }}>{selGene}</div>
        <div style={{ fontSize: 11, color: '#64748b', marginBottom: 12 }}>
          {g.protein_size} · {g.locus} · {g.inheritance} · n={g.n_patients}
        </div>

        {/* Stats bar */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: '1rem' }}>
          {[
            ['Hyperphagia', g.hyperphagia_pct],
            ['Severe Obesity', g.severe_obesity_pct],
            ['Hyperinsulinaemia', g.hyperinsulinaemia_pct],
            ['Tall Stature', g.tall_stature_pct],
            ['Red Hair', g.red_hair_pct],
            ['Adrenal Insuff', g.adrenal_insufficiency_pct],
            ['Hypogonadism', g.hypogonadism_pct],
            ['Neonatal Diarrhoea', g.neonatal_diarrhoea_pct],
            ['Behavioural Issues', g.behavioural_issues_pct],
            ['Low Heart Rate', g.low_heart_rate_pct],
            ['PWL Phenotype', g.pwl_phenotype_pct],
            ['T2D', g.t2d_pct],
            ['Immune Dysfunction', g.immune_dysfunction_pct],
            ['Setmelanotide Eligible', g.setmelanotide_eligible_pct],
          ].map(([lbl, val]) => val > 0 && (
            <div key={lbl} style={{ background: '#0f172a', borderRadius: 6, padding: '4px 8px', fontSize: 11, color: '#94a3b8' }}>
              {lbl}: <span style={{ color: val > 60 ? '#ef4444' : val > 30 ? '#f59e0b' : '#10b981', fontWeight: 600 }}>{val}%</span>
            </div>
          ))}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
          <div>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#94a3b8', marginBottom: 4 }}>Age of Onset</div>
            <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.6 }}>{g.age_of_onset}</div>
          </div>
          <div>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#94a3b8', marginBottom: 4 }}>Key Biomarker</div>
            <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.6 }}>{g.key_biomarker}</div>
          </div>
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#94a3b8', marginBottom: 4 }}>Pathognomonic / Distinguishing Features</div>
          <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.6 }}>{g.pathognomonic}</div>
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#94a3b8', marginBottom: 4 }}>Treatment</div>
          <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.6 }}>{g.treatment}</div>
        </div>

        <div>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#94a3b8', marginBottom: 6 }}>Critical Flags</div>
          {(g.critical_flags || []).map(f => <Alert key={f} text={f} />)}
        </div>
      </div>
    </div>
  );
}

/* ── DEFINITIONS TAB ──────────────────────────────────────────────────────── */
function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h3 style={{ color: '#f1f5f9', marginBottom: '0.5rem' }}>Pathway & Glossary</h3>
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1rem', fontSize: 12, color: '#cbd5e1', lineHeight: 1.7 }}>
        <strong style={{ color: '#a5b4fc' }}>Pathway:</strong> {data.pathway}<br/>
        <strong style={{ color: '#a5b4fc', marginTop: 8, display: 'block' }}>Shared Mechanism:</strong> {data.shared_mechanism}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        {Object.entries(data.glossary || {}).map(([term, def]) => (
          <div key={term} style={{ background: '#1e293b', borderRadius: 8, padding: '0.7rem', fontSize: 11 }}>
            <div style={{ fontWeight: 700, color: '#a5b4fc', marginBottom: 3 }}>{term}</div>
            <div style={{ color: '#94a3b8', lineHeight: 1.5 }}>{def}</div>
          </div>
        ))}
      </div>
      {data.surveillance_protocols && (
        <div style={{ marginTop: '1.5rem' }}>
          <h4 style={{ color: '#f1f5f9', marginBottom: '0.8rem' }}>Surveillance Protocols</h4>
          {Object.entries(data.surveillance_protocols).map(([gene, proto]) => (
            <div key={gene} style={{ background: '#1e293b', borderRadius: 8, padding: '0.6rem 0.8rem', marginBottom: 6, fontSize: 11 }}>
              <span style={{ fontWeight: 700, color: GENE_COLORS[gene] || '#a5b4fc', marginRight: 8 }}>{gene}</span>
              <span style={{ color: '#94a3b8' }}>{proto}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── MAIN PAGE ────────────────────────────────────────────────────────────── */
export default function HereditaryObesityMelanocortinAtlasPage() {
  const [tab, setTab]           = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]         = useState(null);
  const [err, setErr]           = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hereditary-obesity-melanocortin-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-obesity-melanocortin-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-obesity-melanocortin-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefs(df); })
      .catch(e => setErr(e.message));
  }, []);

  return (
    <div style={{ minHeight: '100vh', background: '#0f172a', color: '#e2e8f0', fontFamily: 'system-ui, sans-serif' }}>
      <div style={{ maxWidth: 1100, margin: '0 auto', padding: '2rem 1.5rem' }}>
        <div style={{ marginBottom: '0.3rem', fontSize: 12, color: '#475569' }}>
          Hereditary Disease Atlas · Obesity &amp; Metabolism · Seeds 2006–2013
        </div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: '#f8fafc', marginBottom: '0.3rem' }}>
          Hereditary Obesity Melanocortin Atlas
        </h1>
        <p style={{ margin: '0 0 1.5rem', color: '#94a3b8', fontSize: 13 }}>
          Complete 8-Gene Leptin-Melanocortin Pathway Atlas — MC4R · LEPR · LEP · PCSK1 · POMC · SH2B1 · KSR2 · SIM1
        </p>

        {err && <ErrorBox msg={err} />}

        {/* Tab bar */}
        <div style={{ display: 'flex', gap: 4, marginBottom: '1.5rem', borderBottom: '1px solid #334155', paddingBottom: 4 }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              padding: '6px 16px', borderRadius: '6px 6px 0 0', border: 'none', cursor: 'pointer',
              background: tab === t ? '#6366f1' : 'transparent',
              color: tab === t ? '#fff' : '#64748b', fontSize: 13, fontWeight: tab === t ? 600 : 400,
            }}>{t}</button>
          ))}
        </div>

        {tab === 'Overview'       && <OverviewTab data={overview} />}
        {tab === 'Gene Table'     && <GeneTableTab />}
        {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
        {tab === 'Definitions'    && <DefinitionsTab data={defs} />}
      </div>
    </div>
  );
}
